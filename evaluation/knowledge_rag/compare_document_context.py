"""Paired document-side contextual retrieval on actual MongoDB + Milvus Lite."""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean

from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.document_context import DocumentContextIndex
from deepresearch.knowledge.dual_store import MongoMilvusStore, structured_chunks
from deepresearch.knowledge.embeddings import RetrievalEmbeddingProvider
from deepresearch.knowledge.reranker import CrossEncoderReranker, load_cross_encoder
from .compare_candidate_quota import PolicyView
from .compare_rerank import RecordedReranker
from .metrics_v3 import score
from .run_v3 import load_dataset, percentile

ROOT = Path(__file__).parent
OUT = ROOT/'document_context'
ROUTES = ['plain', 'metadata', 'descriptions']
MODEL = 'thenlper/gte-small-zh'


def summarize(rows):
    result = {}
    for route in ROUTES:
        values = [row['routes'][route] for row in rows]
        keys = [k for k in values[0] if k not in {'hits', 'ms', 'calls', 'channels', 'candidate_availability', 'coarse_availability'}]
        result[route] = {k: mean(v[k] for v in values) for k in keys}
        result[route].update(count=len(values), p50_ms=percentile([v['ms'] for v in values], .5),
                             p95_ms=percentile([v['ms'] for v in values], .95))
    return result


def run(args):
    from milvus_lite.server import Server
    manifest, docs, cases = load_dataset()
    prior = json.loads((ROOT/'candidate_quota/results.json').read_text())
    freeze = json.loads((args.output/'context-freeze.json').read_text())
    context_path = args.output/'contexts.json'
    assert hashlib.sha256(context_path.read_bytes()).hexdigest() == freeze['contexts_sha256']
    assert freeze['protocol']['corpus_sha256'] == manifest['corpus_sha256']
    assert freeze['statuses'] == {'generated': 222}, 'failed descriptions invalidate this comparison'
    contexts = DocumentContextIndex.load(context_path)
    spans = {d['id']: structured_chunks(d['content']) for d in docs}
    chunk_hash = hashlib.sha256(json.dumps(spans, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    old_protocol = prior['protocol']
    assert chunk_hash == old_protocol['chunk_sha256'] and manifest == old_protocol['dataset_manifest']
    protocol = {**old_protocol, 'experiment': 'corpus-only chunk-specific Chinese descriptions',
                'policies': ['quota_dense'], 'order': ROUTES, 'contextual_index': 'varies by route',
                'contexts_sha256': freeze['contexts_sha256'],
                'routes': {'plain': 'original text (+ existing repeated table headers)',
                           'metadata': 'title + heading + original text',
                           'descriptions': 'chunk-specific LLM description + original text; no title/heading prefix'},
                'reranker_input': 'unchanged title + heading + original text, descriptions excluded',
                'freeze': freeze, 'date': '2026-10-20',
                'limits': ['124 development cases; no independent held-out generalization claim.',
                           'Known-span Hit@5, not final answer accuracy or exhaustive Recall.',
                           'Descriptions see the full corresponding document, never queries or labels.',
                           'Plain baseline must exactly reproduce previous quota_dense candidates and hits.',
                           'All routes use the same fixed candidate/reranker/context budgets.',
                           'Single warm CPU observation per route/query; no concurrent load SLA.',
                           'No LLM answer generation; negative nonempty candidates do not measure refusal.']}
    (args.output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    reranker = CrossEncoderReranker(old_protocol['reranker'], timeout=5)
    reranker.model = load_cross_encoder(old_protocol['reranker'], revision=old_protocol['reranker_revision'],
                                       device='cpu', trust_remote_code=False, max_length=512)
    reranker('预热问题', [{'title': '测试', 'content': '这是预热段落。'}])
    provider = RetrievalEmbeddingProvider(MODEL, revision=old_protocol['embedding_revision'], device='cpu')
    provider._load(); provider.encode_query('预热问题')
    report = {'protocol': protocol, 'indexing': {}, 'cases': [], 'summary': {}}
    try:
        with tempfile.TemporaryDirectory(prefix='rag-document-context-') as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            server = Server(str(Path(directory)/'context.db'), address=f'127.0.0.1:{port}')
            assert server.init() and server.start()
            stores, refs, cutoffs, lookups = {}, {}, {}, {}
            try:
                for route in ROUTES:
                    store = MongoMilvusStore(args.mongo_uri, database='rag_context_eval_'+uuid.uuid4().hex,
                        milvus_uri=f'http://127.0.0.1:{port}', provider=provider, candidate_policy='quota_dense',
                        contextual=route == 'metadata', document_context=contexts if route == 'descriptions' else None)
                    stores[route] = store
                    refs[route], cutoffs[route] = {}, {}
                    started = time.perf_counter()
                    for doc in docs:
                        item = store.ingest(doc['content'], uri=doc['uri'], title=doc['title'])
                        refs[route][item['snapshot_id']] = doc['id']
                        version = store.db.versions.find_one({'_id': item['snapshot_id']})
                        cutoffs[route][doc['id']] = version['published_at'].isoformat()+'+00:00'
                        assert version['content'] == doc['content']
                    durations = time.perf_counter()-started
                    versions = {v['_id']: v for v in store.db.versions.find({})}
                    chunks = list(store.db.chunks.find({}))
                    texts = [store._index_text(versions[c['snapshot']], c) for c in chunks]
                    lengths = [len(provider._load().tokenizer.encode(t, truncation=False)) for t in texts]
                    notes = sum(len(v.get('retrieval_contexts', {})) for v in versions.values())
                    assert len(chunks) == 222 and notes == (222 if route == 'descriptions' else 0)
                    assert max(lengths) <= provider._load().max_seq_length, 'added prefix truncates original chunk'
                    report['indexing'][route] = {'seconds': durations, 'chunks': len(chunks),
                        'description_count': notes, 'max_index_tokens': max(lengths),
                        'overflow_count': sum(n > provider._load().max_seq_length for n in lengths)}
                    lookups[route] = {c['_id']: {'document': refs[route][c['snapshot']], 'start': c['start'], 'end': c['end']} for c in chunks}
                    print(f'indexed {route}: {report["indexing"][route]}', flush=True)
                for ci, case in enumerate(cases):
                    row = {k: case[k] for k in ['id', 'query', 'category', 'group']}; row['routes'] = {}
                    for route in ROUTES:
                        store = stores[route]
                        def format_ids(ids): return [lookups[route][cid] for cid in ids]
                        def format_hits(hits):
                            values = []
                            for h in hits:
                                doc = next(d for d in docs if d['id'] == refs[route][h['snapshot_id']])
                                assert h['content'] == doc['content'][h['start']:h['end']]
                                values.append({'document': doc['id'], 'start': h['start'], 'end': h['end'], 'content': h['content']})
                            return values
                        recording = RecordedReranker(reranker, format_hits)
                        store.reranker = recording
                        view = PolicyView(store, 'quota_dense', format_ids)
                        started = time.perf_counter()
                        if case['group'] == 'multi_source':
                            pack, _ = build_context(view, case['subqueries'], limit=5, max_chars=6000, per_source=2)
                            raw = pack['passages']
                        else:
                            kwargs = {'as_of': cutoffs[route][case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                            raw = view.search(case['query'], limit=5, mode='hybrid', **kwargs)
                        elapsed = (time.perf_counter()-started)*1000
                        hits = format_hits(raw)
                        assert recording.calls and all(h['rerank_status'] == 'applied' for h in raw), 'reranker fallback'
                        if route == 'plain':
                            old = prior['cases'][ci]
                            assert old['id'] == case['id']
                            old = old['routes']['quota_dense']
                            assert old['hits'] == hits, f'baseline hit drift: {case["id"]}'
                            assert [(c['query'], c['input']) for c in recording.calls] == [(c['query'], c['input']) for c in old['calls']], 'baseline pool drift'
                            assert old['channels'] == view.channels, 'baseline raw route drift'
                        for call, channel in zip(recording.calls, view.channels):
                            assert call['count'] == len(channel['selected']) <= channel['budget']
                            assert channel['selected'] == channel['validated']
                        pool = [h for call in recording.calls for h in call['input']]
                        coarse = [h for channel in view.channels for h in channel['fused']]
                        row['routes'][route] = {**score(case['evidence_groups'], hits), 'hits': hits, 'ms': elapsed,
                            'candidate_availability': score(case['evidence_groups'], pool, k=len(pool)),
                            'coarse_availability': score(case['evidence_groups'], coarse, k=len(coarse)),
                            'calls': list(recording.calls), 'channels': view.channels}
                    report['cases'].append(row)
                    if (ci+1) % 12 == 0 or ci+1 == len(cases):
                        (args.output/'partial.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                        print(f'document context paired {ci+1}/{len(cases)}', flush=True)
                groups = defaultdict(list)
                for row in report['cases']: groups[row['group']].append(row)
                report['summary'] = {g: summarize(rows) for g, rows in groups.items()}
                (args.output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                (args.output/'partial.json').unlink(missing_ok=True)
                print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
            finally:
                for store in stores.values():
                    store.mongo.drop_database(store.db.name); store.close()
                server.stop()
    finally:
        reranker.executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongo-uri', default='mongodb://127.0.0.1:27028')
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
