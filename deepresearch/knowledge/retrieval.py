"""Shared retrieval logic; storage backends provide snapshots and canonical spans."""
from __future__ import annotations
import io
import json
import math
import re
from numbers import Real
from collections import Counter
from .store import digest, features
from .candidate_selection import POLICIES, quota_targets, select_candidates

def structured_chunks(text, size=600):
    """Preserve contiguous source spans; repeat split-table headers only in index text."""
    result, headings, pending = [], [], None
    blocks = list(re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text))
    for block in blocks:
        start, end = block.span()
        value = block.group()
        first = value.splitlines()[0]
        heading = re.match(r"^(#{1,6})\s+(.+)$", first)
        clause = re.match(r"^(第[零〇一二三四五六七八九十百千万\d]+[编章节条])(?:\s|[、.：:]|$)", first)
        if heading:
            headings = headings[:len(heading[1])-1] + [heading[2]]
        elif clause:
            if clause[1].endswith(('章', '节', '编')):
                headings = [first]
            else:
                headings = headings[:1] + [first]
        if heading and '\n' not in value and len(value) <= size:
            pending = start if pending is None else pending
            continue
        if pending is not None:
            start, pending = pending, None
        table_header = ''
        table_lines = value.splitlines()
        if len(table_lines) > 1 and '|' in table_lines[0] and re.match(r'^\s*\|?\s*:?-+', table_lines[1]):
            table_header = '\n'.join(table_lines[:2])
        cursor = start
        while cursor < end:
            stop = min(cursor+size, end)
            if stop < end:
                marks = '\n' if table_header else '。；.!?\n'
                boundary = max(text.rfind(c, cursor+size//2, stop) for c in marks)
                if boundary >= 0:
                    stop = boundary+1
            result.append({'start': cursor, 'end': stop, 'path': list(headings),
                           'index_prefix': table_header if table_header and cursor > block.start() else ''})
            cursor = stop
    if pending is not None:
        for cursor in range(pending, len(text), size):
            result.append({'start': cursor, 'end': min(cursor+size, len(text)),
                           'path': list(headings), 'index_prefix': ''})
    return result


def validate_vectors(vectors, count, dimensions):
    if len(vectors) != count:
        raise ValueError('incomplete embedding batch')
    for vector in vectors:
        if len(vector) != dimensions or dimensions <= 0:
            raise ValueError('embedding dimension mismatch')
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in vector):
            raise ValueError('embedding must contain finite numeric values')
        if not any(v != 0 for v in vector):
            raise ValueError('zero embedding is not ready')



class DocumentRetrievalMixin:
    def _index_projection(self, content, parts):
        notes = self.document_context.projection(content, parts) if self.document_context else {}
        return {'index_context': self.contextual, 'retrieval_contexts': notes,
                'context_manifest_sha256': self.document_context.fingerprint if notes else None}


    def _index_text(self, version, row):
        text = version['content'][row['start']:row['end']]
        prefix = row.get('index_prefix', '')
        # Use the published projection, not a reader's current feature switches.
        if version.get('index_context', self.contextual):
            prefix = '\n'.join([version['title'][:200], ' > '.join(row['heading'])[:300], prefix])
        note = version.get('retrieval_contexts', {}).get(f"{row['start']}:{row['end']}")
        if note:
            if digest(text) != note['chunk_sha256']:
                raise ValueError('retrieval description source hash mismatch')
            prefix = '\n'.join(filter(None, [prefix, note['text']]))
        return prefix+'\n'+text if prefix else text


    def _parse(self, data, suffix):
        if self.parser == 'mineru' and suffix == '.pdf':
            from .parsing import mineru_parse
            return mineru_parse(data)
        if suffix == '.pdf':
            from .document_parser import parse_pdf
            return parse_pdf(data, options=getattr(self, 'parse_options', None),
                             native_only=self.parser == 'native')
        if suffix in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
            if self.parser == 'native':
                raise ValueError('Images require parser=auto and a local OCR backend')
            from .document_parser import parse_image
            return parse_image(data, suffix, options=getattr(self, 'parse_options', None))
        if suffix in ['.md', '.txt']:
            return data.decode('utf-8'), {'parser': 'utf8'}
        if suffix == '.docx':
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            blocks = []
            for block in Document(io.BytesIO(data)).iter_inner_content():
                if isinstance(block, Paragraph):
                    style = block.style.name or ''
                    match = re.match('Heading ([1-6])', style)
                    blocks.append(('#'*int(match[1])+' ' if match else '')+block.text)
                elif isinstance(block, Table):
                    rows = [[cell.text.replace('\n',' ').replace('|','\\|') for cell in row.cells] for row in block.rows]
                    if rows:
                        blocks.append('\n'.join(['| '+' | '.join(rows[0])+' |', '| '+' | '.join(['---']*len(rows[0]))+' |',
                                       *['| '+' | '.join(row)+' |' for row in rows[1:]]]))
            return '\n\n'.join(blocks), {'parser': 'python-docx'}
        raise ValueError('supported formats: .md, .txt, .pdf, .docx, .png, .jpg, .jpeg, .tif, .tiff, .bmp')


    def search(self, query, *, collection='library', snapshot_id=None, as_of=None, limit=5, mode='hybrid', rerank=True,
               candidate_policy=None, trace=None):
        if mode not in ['hybrid', 'bm25', 'vector']:
            raise ValueError('unknown retrieval mode')
        requested_policy = self.candidate_policy if candidate_policy is None else candidate_policy
        if requested_policy not in POLICIES:
            raise ValueError('unknown candidate policy')
        policy = requested_policy if mode == 'hybrid' else 'rrf'
        if not query.strip() or limit <= 0:
            return []
        versions = self._eligible(collection, snapshot_id, as_of)
        if not versions:
            return []
        by_sid = {v['_id']: v for v in versions}
        rows = self._chunk_rows(versions)
        # Stable ties across rebuilds/models; random staging IDs cannot decide relevance.
        def tie(cid):
            row = by_id[cid]
            version = by_sid[row['snapshot']]
            return version['document_id'], row['start'], row['end'], version['hash']
        by_id = {row['_id']: row for row in rows}
        lexical, semantic = [], []
        if mode != 'vector':
            counts = [Counter(features(self._index_text(by_sid[row['snapshot']], row))) for row in rows]
            terms = set(features(query)); df = Counter(t for c in counts for t in c if t in terms)
            lengths = [sum(c.values()) for c in counts]; avg = sum(lengths)/max(1,len(lengths)) or 1
            for row, freq, length in zip(rows, counts, lengths, strict=True):
                score = sum(math.log(1+(len(rows)-df[t]+.5)/(df[t]+.5))*freq[t]*2.2 /
                    (freq[t]+1.2*(.25+.75*length/avg)) for t in terms if freq[t])
                if score > 0:
                    lexical.append((row['_id'], score))
            lexical.sort(key=lambda x: (-x[1], tie(x[0]))); lexical = lexical[:20]
        vector_status = 'disabled' if mode == 'bm25' else 'available'
        if mode != 'bm25':
            try:
                with self.lock:
                    vector = self.provider.encode_query(query)
                    validate_vectors([vector], 1, self.provider.dimensions)
                    if self.provider.degraded:
                        raise ValueError('semantic embedding unavailable')
                    model = self.provider.model_id
                names = {v['vector_collection'] for v in versions if v.get('model') == model}
                for name in names:
                    sids = [v['_id'] for v in versions if v.get('vector_collection') == name]
                    status_filter = 'status in ["active", "archived"]' if snapshot_id or as_of else 'status == "active"'
                    filt = 'scope == '+json.dumps(digest(collection))+' and '+status_filter+' and snapshot in '+json.dumps(sids)
                    hits = self.vectors.search(collection_name=name, data=[vector], filter=filt, limit=40,
                        output_fields=['snapshot'], consistency_level='Strong')[0]
                    for hit in hits:
                        if hit['id'] in by_id:
                            semantic.append((hit['id'], float(hit['distance'])))
                semantic.sort(key=lambda x: (-x[1], tie(x[0]))); semantic = semantic[:40]
            except Exception:
                vector_status = 'fallback_bm25'
                if mode == 'vector':
                    raise
        scores, route_ids = {}, []
        for route in ([lexical] if mode == 'bm25' else [semantic] if mode == 'vector' else [lexical, semantic]):
            seen, ids = set(), []
            for rank, (cid, _) in enumerate(route,1):
                row = by_id[cid]; version = by_sid[row['snapshot']]
                key = (row['snapshot'], digest(version['content'][row['start']:row['end']].strip()))
                if key in seen:
                    continue
                seen.add(key); scores[cid] = scores.get(cid,0)+1/(60+rank)
                ids.append(cid)
            route_ids.append(ids)
        budget = max(20,limit*2)
        fused = sorted(scores, key=lambda cid: (-scores[cid], tie(cid)))
        selected = select_candidates(route_ids[0] if mode != 'vector' else [],
            route_ids[-1] if mode != 'bm25' else [], fused, budget=budget, policy=policy)
        if trace is not None:
            lexical_target, semantic_target = quota_targets(budget, policy)
            trace.update(policy=policy, budget=budget, lexical_quota=lexical_target, semantic_quota=semantic_target,
                lexical=route_ids[0] if mode != 'vector' else [],
                semantic=route_ids[-1] if mode != 'bm25' else [], fused=fused, selected=selected,
                vector_status=vector_status)
        hits = []
        for cid in selected:
            row = by_id[cid]
            # Recheck the published source version; retrieval projections are not evidence.
            version = self.snapshot(row['snapshot'], collection=collection)
            if not snapshot_id and not as_of and version['status'] != 'active':
                continue
            text = version['content'][row['start']:row['end']]
            pages = json.loads(version['metadata']).get('page_starts', [])
            hits.append({'chunk_id': cid, 'snapshot_id': version['_id'], 'uri': version['uri'],
                'title': version['title'], 'content': text, 'start': row['start'], 'end': row['end'],
                'heading': row['heading'], 'table_header': row.get('index_prefix') or None,
                'page': sum(o<=row['start'] for o in pages) if pages else None, 'score': scores[cid],
                'degraded': bool(version.get('degraded')) or vector_status == 'fallback_bm25',
                'model_mismatch': version.get('model') != self.provider.model_id, 'index_context': bool(version.get('index_context')),
                'rerank_status': 'disabled', 'vector_status': vector_status})
        if trace is not None:
            trace['validated'] = [h['chunk_id'] for h in hits]
        if rerank and self.reranker and hits:
            try:
                ranked = self.reranker(query, self.prepare_rerank_hits(hits, collection=collection))
                if sorted(h['chunk_id'] for h in ranked) != sorted(h['chunk_id'] for h in hits):
                    raise ValueError('reranker must return a permutation')
                trusted = {h['chunk_id']: h for h in hits}; hits = [trusted[h['chunk_id']] for h in ranked]
                for hit in hits:
                    hit['rerank_status'] = 'applied'
            except Exception:
                for hit in hits:
                    hit['rerank_status'] = 'fallback'
        return hits[:limit]


    def prepare_rerank_hits(self, hits, *, collection='library'):
        """Bounded original neighbors for scoring only; never enlarge evidence spans."""
        inputs, snapshots = [], {}
        for hit in hits:
            value = dict(hit)
            value.pop('_rerank_neighbors', None)
            if self.reranker_neighbor_chars:
                sid = hit['snapshot_id']
                if sid not in snapshots:
                    snapshots[sid] = self.snapshot(sid, collection=collection)
                content = snapshots[sid]['content']
                a, b = hit['start'], hit['end']
                if content[a:b] != hit['content']:
                    raise ValueError('reranker target differs from immutable source')
                radius = self.reranker_neighbor_chars
                value['_rerank_neighbors'] = {'before': content[max(0,a-radius):a], 'after': content[b:b+radius]}
            inputs.append(value)
        return inputs
