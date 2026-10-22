"""Fixed-candidate reranking of the frozen document-context index, three full routes."""
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
from deepresearch.knowledge.dual_store import MongoMilvusStore
from deepresearch.knowledge.embeddings import RetrievalEmbeddingProvider
from deepresearch.knowledge.reranker import CrossEncoderReranker, document_for_rerank, load_cross_encoder
from .compare_candidate_quota import PolicyView
from .metrics_v3 import score
from .run_v3 import load_dataset, percentile

ROOT = Path(__file__).parent
OUT = ROOT/'context_rerank'
MINI = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
BGE = 'BAAI/bge-reranker-v2-m3'
ROUTES = {'mini_original': {'model': MINI, 'neighbors': 0},
          'mini_neighbors': {'model': MINI, 'neighbors': 120},
          'bge_original': {'model': BGE, 'neighbors': 0},
          'bge_neighbors': {'model': BGE, 'neighbors': 120}}
DEFAULT_ROUTES = list(ROUTES)[:3]


def summarize(rows, routes=None):
    result = {}
    for route in routes or ROUTES:
        values = [row['routes'][route] for row in rows]
        metrics = [k for k in values[0] if k not in {'hits', 'ms', 'calls', 'channels', 'candidate_availability'}]
        result[route] = {k: mean(v[k] for v in values) for k in metrics}
        result[route].update(count=len(values), p50_ms=percentile([v['ms'] for v in values], .5),
                             p95_ms=percentile([v['ms'] for v in values], .95))
    return result


class RecordedInputs:
    def __init__(self, reranker, format_hits):
        self.reranker, self.format_hits, self.calls = reranker, format_hits, []

    def __call__(self, query, hits):
        tokenizer = self.reranker.model.tokenizer
        texts = [document_for_rerank(h) for h in hits]
        lengths = [len(tokenizer(query, text, truncation=False)['input_ids']) for text in texts]
        target_lengths = [len(tokenizer(query, document_for_rerank({k:v for k,v in h.items() if k != '_rerank_neighbors'}),
                                        truncation=False)['input_ids']) for h in hits]
        # The target is always first; only optional neighbor tails may overflow.
        assert max(target_lengths) <= 512, 'reranker truncates original target'
        started = time.perf_counter()
        ranked = self.reranker(query, hits)
        elapsed = (time.perf_counter()-started)*1000
        self.calls.append({'query': query, 'count': len(hits), 'input': self.format_hits(hits),
            'output': self.format_hits(ranked), 'rerank_ms': elapsed,
            'scoring_text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in texts],
            'pair_tokens': lengths, 'target_pair_tokens': target_lengths,
            'neighbor_char_counts': [{k:len(v) for k,v in h.get('_rerank_neighbors', {}).items()} for h in hits]})
        return ranked


def run(args):
    from milvus_lite.server import Server
    manifest, docs, cases = load_dataset()
    prior = json.loads((ROOT/'document_context/results.json').read_text())
    old = prior['protocol']
    context_path = ROOT/'document_context/contexts.json'
    assert manifest == old['dataset_manifest']
    assert hashlib.sha256(context_path.read_bytes()).hexdigest() == old['contexts_sha256']
    bge_prior = json.loads((ROOT/'reranker_models/bge_v2_m3/results.json').read_text())
    revisions = {MINI: old['reranker_revision'], BGE: bge_prior['reranker_revision']}
    routes = {name: ROUTES[name] for name in args.routes}
    protocol = {'date': '2026-10-20', 'dataset_manifest': manifest, 'contexts_sha256': old['contexts_sha256'],
        'embedding': old['embedding'], 'embedding_revision': old['embedding_revision'],
        'routes': routes, 'model_revisions': revisions, 'order': list(routes), 'device': 'cpu',
        'max_reranker_tokens': 512, 'reranker_timeout_seconds': 30,
        'production_default_timeout_seconds': 5, 'candidate_policy': 'quota_dense',
        'bm25': 20, 'vector': 40, 'rrf_k': 60, 'regular_rerank_pool': 20, 'worker_subquery_pool': 40,
        'context': {'limit': 5, 'max_chars': 6000, 'per_source': 2},
        'query_rewrite': False, 'generated_descriptions_in_reranker': False,
        'neighbor_layout': 'original target first, then up to120 source characters before/after; no evidence span expansion',
        'limits': ['Same 124 development questions, no independent held-out generalization claim.',
                   'All routes have exactly the same ordered candidate IDs/source spans, verified against previous context experiment.',
                   'Known-span Hit@5 is not exhaustive Recall or answer accuracy.',
                   '30s experimental reranking budget; report calls exceeding production default5s separately.',
                   'Warm CPU single observation with fixed route order, not a production SLA.',
                   'No additional LLM generation, no changes to frozen descriptions or labels.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    rankers, report = {}, {'protocol': protocol, 'cases': [], 'summary': {}, 'model_load_seconds': {}}
    try:
        for name in dict.fromkeys(config['model'] for config in routes.values()):
            started = time.perf_counter()
            ranker = CrossEncoderReranker(name, timeout=30)
            rankers[name] = ranker
            ranker.model = load_cross_encoder(name, revision=revisions[name], device='cpu',
                                              trust_remote_code=False, max_length=512)
            ranker('预热问题', [{'title': '测试', 'content': '这是预热段落。'}])
            report['model_load_seconds'][name] = time.perf_counter()-started
        provider = RetrievalEmbeddingProvider(old['embedding'], revision=old['embedding_revision'], device='cpu')
        provider._load(); provider.encode_query('预热问题')
        with tempfile.TemporaryDirectory(prefix='rag-context-rerank-') as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            server = Server(str(Path(directory)/'rerank.db'), address=f'127.0.0.1:{port}')
            assert server.init() and server.start()
            store = None
            try:
                store = MongoMilvusStore(args.mongo_uri, database='rag_context_rerank_'+uuid.uuid4().hex,
                    milvus_uri=f'http://127.0.0.1:{port}', provider=provider, candidate_policy='quota_dense',
                    document_context=DocumentContextIndex.load(context_path))
                refs, cutoffs = {}, {}
                for doc in docs:
                    item = store.ingest(doc['content'], uri=doc['uri'], title=doc['title'])
                    refs[item['snapshot_id']] = doc['id']
                    version = store.db.versions.find_one({'_id': item['snapshot_id']})
                    cutoffs[doc['id']] = version['published_at'].isoformat()+'+00:00'
                docmap = {d['id']: d for d in docs}
                lookup = {c['_id']: {'document': refs[c['snapshot']], 'start': c['start'], 'end': c['end']} for c in store.db.chunks.find({})}
                def format_ids(ids): return [lookup[cid] for cid in ids]
                def format_hits(hits):
                    values = []
                    for h in hits:
                        did = refs[h['snapshot_id']]
                        assert h['content'] == docmap[did]['content'][h['start']:h['end']]
                        values.append({'document': did, 'start': h['start'], 'end': h['end'], 'content': h['content']})
                    return values
                for ci, case in enumerate(cases):
                    row = {k: case[k] for k in ['id', 'query', 'category', 'group']}; row['routes'] = {}
                    expected = prior['cases'][ci]
                    assert expected['id'] == case['id']
                    expected = expected['routes']['descriptions']
                    for route, config in routes.items():
                        store.reranker_neighbor_chars = config['neighbors']
                        recording = RecordedInputs(rankers[config['model']], format_hits)
                        store.reranker = recording
                        view = PolicyView(store, 'quota_dense', format_ids)
                        started = time.perf_counter()
                        if case['group'] == 'multi_source':
                            pack, _ = build_context(view, case['subqueries'], limit=5, max_chars=6000, per_source=2)
                            raw = pack['passages']
                        else:
                            kwargs = {'as_of': cutoffs[case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                            raw = view.search(case['query'], limit=5, mode='hybrid', **kwargs)
                        elapsed = (time.perf_counter()-started)*1000
                        assert recording.calls and all(h['rerank_status'] == 'applied' for h in raw)
                        assert all('_rerank_neighbors' not in h for h in raw), 'private scoring context leaked'
                        hits = format_hits(raw)
                        assert view.channels == expected['channels'], 'raw candidate route drift'
                        assert [(c['query'], c['input']) for c in recording.calls] == [(c['query'], c['input']) for c in expected['calls']], 'ordered candidate drift'
                        if route == 'mini_original':
                            assert hits == expected['hits'], 'baseline hit drift'
                        pool = [h for call in recording.calls for h in call['input']]
                        row['routes'][route] = {**score(case['evidence_groups'], hits), 'hits': hits, 'ms': elapsed,
                            'candidate_availability': score(case['evidence_groups'], pool, k=len(pool)),
                            'calls': recording.calls, 'channels': view.channels}
                    report['cases'].append(row)
                    if (ci+1) % 6 == 0 or ci+1 == len(cases):
                        (args.output/'partial.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                        print(f'context rerank paired {ci+1}/{len(cases)}', flush=True)
                groups = defaultdict(list)
                for row in report['cases']: groups[row['group']].append(row)
                report['summary'] = {g: summarize(rows, routes) for g, rows in groups.items()}
                (args.output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                (args.output/'partial.json').unlink(missing_ok=True)
                print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
            finally:
                if store is not None:
                    store.mongo.drop_database(store.db.name); store.close()
                server.stop()
    finally:
        for ranker in rankers.values(): ranker.executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongo-uri', default='mongodb://127.0.0.1:27028')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--routes', nargs='+', choices=list(ROUTES), default=DEFAULT_ROUTES)
    run(parser.parse_args())
