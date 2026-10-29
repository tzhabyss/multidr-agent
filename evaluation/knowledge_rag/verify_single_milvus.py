"""Re-run the frozen 124-case balanced route on the single-Milvus backend.

No new labels, rewrites, LLM calls or parameter tuning. Compare source spans to
historical dual-store bge_original results; these remain development metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import socket
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.document_context import DocumentContextIndex
from deepresearch.knowledge.embeddings import RetrievalEmbeddingProvider
from deepresearch.knowledge.milvus_store import MilvusStore
from deepresearch.knowledge.reranker import CrossEncoderReranker, load_cross_encoder
from .metrics_v3 import score
from .run_v3 import load_dataset, percentile

ROOT = Path(__file__).parent


def run(output):
    from milvus_lite.server import Server
    manifest, docs, cases = load_dataset()
    prior = json.loads((ROOT/'context_rerank/results.json').read_text())
    protocol = prior['protocol']
    contexts = ROOT/'document_context/contexts.json'
    assert manifest == protocol['dataset_manifest']
    assert hashlib.sha256(contexts.read_bytes()).hexdigest() == protocol['contexts_sha256']
    output.mkdir(parents=True, exist_ok=True)
    provider = RetrievalEmbeddingProvider(protocol['embedding'], revision=protocol['embedding_revision'], device='cpu')
    ranker = CrossEncoderReranker('BAAI/bge-reranker-v2-m3', timeout=30)
    ranker.model = load_cross_encoder(ranker.name, revision=protocol['model_revisions'][ranker.name],
        device='cpu', trust_remote_code=False, max_length=512)
    provider._load(); provider.encode_query('预热')
    ranker('预热', [{'title': '测试', 'content': '预热段落'}])
    report = {'protocol': {**protocol, 'backend': 'milvus', 'route': 'bge_original',
        'comparison': 'historical frozen dual-store results (not a fresh paired latency benchmark)',
        'retrieval_parameters_changed': False, 'source': 'real Milvus Lite 2.5',
        'limits': ['124 existing development cases, not held-out tests or answer accuracy.',
                   'Negative cases measure nonempty candidates, not refusal.',
                   'Warm CPU single-run latency, no production performance claim.']},
        'cases': [], 'summary': {}, 'historical_differences': []}
    try:
        with tempfile.TemporaryDirectory(prefix='single-milvus-eval-') as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            server = Server(str(Path(directory)/'eval.db'), address=f'127.0.0.1:{port}')
            assert server.init() and server.start()
            store = None
            try:
                store = MilvusStore(milvus_uri=f'http://127.0.0.1:{port}', lock_dir=directory,
                    provider=provider, reranker=ranker, candidate_policy='quota_dense',
                    document_context=DocumentContextIndex.load(contexts))
                refs, cutoffs = {}, {}
                start = time.perf_counter()
                for doc in docs:
                    item = store.ingest(doc['content'], uri=doc['uri'], title=doc['title'])
                    refs[item['snapshot_id']] = doc['id']
                    cutoffs[doc['id']] = store.snapshot(item['snapshot_id'], collection='library')['published_at']
                report['index_seconds'] = time.perf_counter()-start
                report['chunks'] = sum(v['expected_chunks'] for v in store.records.list('version'))
                assert report['chunks'] == 222
                docmap = {d['id']: d for d in docs}
                def format_hits(raw):
                    result = []
                    for hit in raw:
                        did = refs[hit['snapshot_id']]
                        assert hit['content'] == docmap[did]['content'][hit['start']:hit['end']]
                        assert not hit['degraded'] and hit['rerank_status'] == 'applied'
                        result.append({'document': did, 'start': hit['start'], 'end': hit['end'], 'content': hit['content']})
                    return result
                for i, case in enumerate(cases):
                    start = time.perf_counter()
                    if case['group'] == 'multi_source':
                        pack, _ = build_context(store, case['subqueries'], limit=5, max_chars=6000, per_source=2)
                        raw = pack['passages']
                    else:
                        kwargs = {'as_of': cutoffs[case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                        raw = store.search(case['query'], limit=5, **kwargs)
                    elapsed = (time.perf_counter()-start)*1000
                    hits = format_hits(raw)
                    old = prior['cases'][i]
                    assert old['id'] == case['id']
                    expected = old['routes']['bge_original']['hits']
                    if hits != expected:
                        report['historical_differences'].append(case['id'])
                    report['cases'].append({'id': case['id'], 'group': case['group'], 'query': case['query'],
                        **score(case['evidence_groups'], hits), 'hits': hits, 'ms': elapsed,
                        'same_ordered_spans_as_historical': hits == expected})
                    if (i+1) % 12 == 0 or i+1 == len(cases):
                        (output/'partial.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                        print(f'single Milvus: {i+1}/{len(cases)}', flush=True)
                groups = defaultdict(list)
                for row in report['cases']:
                    groups[row['group']].append(row)
                for group, rows in groups.items():
                    keys = ['candidate_nonempty'] if group == 'negative' else [
                        'hit_at_5', 'mrr_at_5', 'evidence_group_coverage_at_5', 'all_groups_at_5']
                    report['summary'][group] = {k: mean(r[k] for r in rows) for k in keys}
                    report['summary'][group].update(count=len(rows), p50_ms=percentile([r['ms'] for r in rows], .5),
                                                    p95_ms=percentile([r['ms'] for r in rows], .95))
                report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                    Path('deepresearch/knowledge/milvus_store.py'), Path('deepresearch/knowledge/milvus_records.py'),
                    Path('deepresearch/knowledge/retrieval.py'), Path(__file__)]}
                (output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                (output/'partial.json').unlink(missing_ok=True)
                print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
                print('Historical ordered-span differences:', len(report['historical_differences']), flush=True)
            finally:
                if store:
                    store.close()
                server.stop()
    finally:
        ranker.executor.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'single_milvus')
    run(parser.parse_args().output)
