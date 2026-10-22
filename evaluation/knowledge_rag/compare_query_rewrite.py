"""Frozen query-only rewriting, fixed-budget fusion and expanded-budget controls."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import socket
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean

from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.dual_store import MongoMilvusStore, structured_chunks
from deepresearch.knowledge.embeddings import RetrievalEmbeddingProvider
from deepresearch.knowledge.query_rewrite import PROMPT_SHA256, RewrittenQuerySearch
from deepresearch.knowledge.reranker import CrossEncoderReranker, load_cross_encoder
from .compare_rerank import RecordedReranker
from .metrics_v3 import score
from .run_v3 import load_dataset, percentile

ROOT = Path(__file__).parent
OUT = ROOT/'query_rewrite'
MODEL = 'thenlper/gte-small-zh'
ROUTES = ['original', 'original_expanded', 'rewrite_rrf', 'rewrite_union']


def summarize(rows):
    result = {}
    for route in ROUTES:
        values = [row['routes'][route] for row in rows]
        keys = [k for k in values[0] if k not in {'hits', 'ms', 'calls', 'generation_ms', 'end_to_end_ms', 'candidate_availability'}]
        result[route] = {k: mean(v[k] for v in values) for k in keys}
        result[route].update(count=len(values),
            p50_ms=percentile([v['ms'] for v in values], .5),
            p95_ms=percentile([v['ms'] for v in values], .95),
            end_to_end_p50_ms=percentile([v['end_to_end_ms'] for v in values], .5),
            end_to_end_p95_ms=percentile([v['end_to_end_ms'] for v in values], .95))
    return result


def run(args):
    from milvus_lite.server import Server
    manifest, docs, cases = load_dataset()
    prior = json.loads((ROOT/'rerank_comparison/results.json').read_text())
    frozen = json.loads((args.output/'rewrite-freeze.json').read_text())
    rewrite_path = args.output/'rewrites.json'
    assert hashlib.sha256(rewrite_path.read_bytes()).hexdigest() == frozen['rewrites_sha256']
    rewrites = json.loads(rewrite_path.read_text())
    assert rewrites['protocol'] == frozen['protocol']
    assert rewrites['protocol']['prompt_sha256'] == PROMPT_SHA256
    assert rewrites['protocol']['cases_sha256'] == manifest['cases_sha256']
    questions = set(q for c in cases for q in (c['subqueries'] if c['group']=='multi_source' else [c['query']]))
    assert set(rewrites['queries']) == questions
    spans = {d['id']: structured_chunks(d['content']) for d in docs}
    chunk_hash = hashlib.sha256(json.dumps(spans, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert prior['chunk_sha256'] == chunk_hash and prior['dataset_manifest'] == manifest
    # Written before any model is loaded or scored. No scoring-dependent rewrites.
    protocol = {'dataset_manifest': manifest, 'chunk_sha256': chunk_hash,
        'rewrite_sha256': frozen['rewrites_sha256'], 'rewrite_protocol': rewrites['protocol'],
        'embedding': MODEL, 'embedding_revision': prior['models'][MODEL]['embedding_revision'],
        'reranker': prior['reranker'], 'reranker_revision': prior['reranker_revision'],
        'device': 'cpu', 'max_reranker_tokens': 512, 'bm25_top_k': 20, 'vector_top_k': 40, 'rrf_k': 60,
        'routes': {'original': 'Original query; regular pool20, Worker subquery pool40',
                   'original_expanded': 'Original query only; double the pool cap (40/80)',
                   'rewrite_rrf': 'Original and rewritten query; each pool20/40, cross-query RRF, final pool cap20/40',
                   'rewrite_union': 'Keep entire original pool20/40 and append novel rewrite pool20/40 hits; cap40/80'},
        'order': ROUTES, 'reranker_timeout_seconds': 5,
        'context': {'limit': 5, 'max_chars': 6000, 'per_source': 2},
        'end_to_end_time': 'Measured local search time + saved query-only generation time; not a live uncached system replay.',
        'limits': ['Frozen development set, not a blind test; known-span Hit@5 is not exhaustive Recall or answer accuracy.',
                   'Single warm CPU observation per route/query; fixed route order, not an SLA.',
                   'Generation latencies were measured with four concurrent API requests.',
                   'LLM failures/unchanged rewrites retain the original query and remain in all denominators.',
                   'Evaluation context excludes production usage-summary telemetry; same 6000-char budget on all routes.'],
        'environment': {k: importlib.metadata.version(k) for k in ['pymongo','pymilvus','milvus-lite','sentence-transformers','torch']}}
    (args.output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    reranker = CrossEncoderReranker(prior['reranker'], timeout=5)
    reranker.model = load_cross_encoder(prior['reranker'], revision=prior['reranker_revision'],
                                       device='cpu', trust_remote_code=False, max_length=512)
    reranker('预热问题', [{'title':'测试','content':'这是预热段落。'}])
    provider = RetrievalEmbeddingProvider(MODEL, revision=protocol['embedding_revision'], device='cpu')
    provider._load(); provider.encode_query('预热问题')
    report = {'protocol': protocol, 'cases': [], 'summary': {}}
    try:
        with tempfile.TemporaryDirectory(prefix='rag-query-rewrite-') as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            server = Server(str(Path(directory)/'rewrite.db'), address=f'127.0.0.1:{port}')
            assert server.init() and server.start()
            try:
                store = MongoMilvusStore(args.mongo_uri, database='rag_rewrite_eval_'+uuid.uuid4().hex,
                    milvus_uri=f'http://127.0.0.1:{port}', provider=provider)
                try:
                    refs, cutoffs = {}, {}
                    for doc in docs:
                        item = store.ingest(doc['content'], uri=doc['uri'], title=doc['title'])
                        refs[item['snapshot_id']] = doc['id']
                        cutoffs[doc['id']] = store.db.versions.find_one({'_id': item['snapshot_id']})['published_at'].isoformat()+'+00:00'
                    def format_hits(hits):
                        return [{'document': refs[h['snapshot_id']], 'start': h['start'], 'end': h['end'], 'content': h['content']} for h in hits]
                    raw_search = store.search
                    def checked_search(*args, **kwargs):
                        hits = raw_search(*args, **kwargs)
                        if any(h['vector_status']!='available' for h in hits):
                            raise RuntimeError('vector fallback invalidates comparison')
                        return hits
                    store.search = checked_search
                    recording = RecordedReranker(reranker, format_hits)
                    store.reranker = recording
                    valid = {q: r['rewritten_query'] for q, r in rewrites['queries'].items() if r['status']=='rewritten'}
                    for ci, case in enumerate(cases):
                        row = {k: case[k] for k in ['id','query','category','group']}; row['routes'] = {}
                        active_queries = case['subqueries'] if case['group']=='multi_source' else [case['query']]
                        row['rewrites'] = [rewrites['queries'][q] for q in active_queries]
                        for route in ROUTES:
                            view = store if route=='original' else RewrittenQuerySearch(store,
                                valid if route.startswith('rewrite_') else {},
                                fusion='anchored_union' if route=='rewrite_union' else 'rrf',
                                multiplier=2 if route=='original_expanded' else 1, record=True)
                            recording.calls = []
                            started = time.perf_counter()
                            if case['group']=='multi_source':
                                pack, _ = build_context(view, active_queries, limit=5, max_chars=6000, per_source=2)
                                raw = pack['passages']
                            else:
                                kwargs = {'as_of': cutoffs[case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                                raw = view.search(case['query'], limit=5, mode='hybrid', **kwargs)
                            elapsed = (time.perf_counter()-started)*1000
                            hits = format_hits(raw)
                            if not recording.calls or any(h['rerank_status']!='applied' for h in raw):
                                raise RuntimeError(f'rerank fallback {case["id"]}/{route}')
                            if route=='original':
                                old = prior['models'][MODEL]['cases'][ci]
                                if old['id']!=case['id'] or old['routes']['on']['hits']!=hits:
                                    raise ValueError(f'baseline drift {case["id"]}')
                                if [(c['query'], c['input']) for c in recording.calls] != [(c['query'], c['input']) for c in old['rerank_calls']]:
                                    raise ValueError('baseline candidate drift')
                            else:
                                for call in view.calls:
                                    if call['rewrite_error'] or any(h['vector_status']!='available' for h in [*call['original'], *call['alternate']]):
                                        raise RuntimeError('retrieval fallback invalidates rewrite comparison')
                            generation_ms = max((rewrites['queries'][q]['ms'] for q in active_queries), default=0) if route.startswith('rewrite_') else 0
                            pool = [h for c in recording.calls for h in c['input']]
                            availability = score(case['evidence_groups'], pool, k=len(pool))
                            row['routes'][route] = {**score(case['evidence_groups'], hits), 'hits': hits,
                                'ms': elapsed, 'generation_ms': generation_ms, 'end_to_end_ms': elapsed+generation_ms,
                                'candidate_availability': availability, 'calls': list(recording.calls)}
                        report['cases'].append(row)
                        if (ci+1)%12==0 or ci+1==len(cases):
                            (args.output/'partial.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                            print(f'query rewrite paired {ci+1}/{len(cases)}', flush=True)
                    groups = defaultdict(list)
                    for row in report['cases']: groups[row['group']].append(row)
                    report['summary'] = {g: summarize(rows) for g, rows in groups.items()}
                    (args.output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                    (args.output/'partial.json').unlink(missing_ok=True)
                    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
                finally:
                    store.mongo.drop_database(store.db.name); store.close()
            finally:
                server.stop()
    finally:
        reranker.executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongo-uri', default='mongodb://127.0.0.1:27028')
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
