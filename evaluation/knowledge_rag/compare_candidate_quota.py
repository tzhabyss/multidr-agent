"""Fixed-budget per-channel reservations on the actual Mongo/Milvus search path."""
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

from deepresearch.knowledge.candidate_selection import POLICIES
from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.dual_store import MongoMilvusStore, structured_chunks
from deepresearch.knowledge.embeddings import RetrievalEmbeddingProvider
from deepresearch.knowledge.reranker import CrossEncoderReranker, load_cross_encoder
from .compare_rerank import RecordedReranker
from .metrics_v3 import score
from .run_v3 import load_dataset, percentile

ROOT = Path(__file__).parent
OUT = ROOT/'candidate_quota'
MODEL = 'thenlper/gte-small-zh'
ROUTES = list(POLICIES)


def summarize(rows):
    result = {}
    for route in ROUTES:
        values = [row['routes'][route] for row in rows]
        keys = [k for k in values[0] if k not in {'hits','ms','calls','channels','candidate_availability'}]
        result[route] = {k:mean(v[k] for v in values) for k in keys}
        result[route].update(count=len(values), p50_ms=percentile([v['ms'] for v in values],.5),
                             p95_ms=percentile([v['ms'] for v in values],.95))
    return result


class PolicyView:
    def __init__(self, store, policy, format_ids):
        self.store, self.policy, self.format_ids = store, policy, format_ids
        self.channels = []
    def snapshot(self, *args, **kwargs):
        return self.store.snapshot(*args, **kwargs)
    def search(self, query, **kwargs):
        trace = {}
        hits = self.store.search(query, candidate_policy=self.policy, trace=trace, **kwargs)
        if trace.get('vector_status')!='available' or any(h['vector_status']!='available' for h in hits):
            raise RuntimeError('vector fallback invalidates quota comparison')
        entry = {**trace, 'query':query}
        for key in ['lexical','semantic','fused','selected','validated']:
            entry[key] = self.format_ids(trace[key])
        self.channels.append(entry)
        return hits


def run(args):
    from milvus_lite.server import Server
    manifest, docs, cases = load_dataset()
    prior = json.loads((ROOT/'rerank_comparison/results.json').read_text())
    spans = {d['id']:structured_chunks(d['content']) for d in docs}
    chunk_hash = hashlib.sha256(json.dumps(spans,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    assert prior['dataset_manifest']==manifest and prior['chunk_sha256']==chunk_hash
    protocol = {'dataset_manifest':manifest, 'chunk_sha256':chunk_hash,
        'embedding':MODEL, 'embedding_revision':prior['models'][MODEL]['embedding_revision'],
        'reranker':prior['reranker'], 'reranker_revision':prior['reranker_revision'],
        'device':'cpu', 'max_reranker_tokens':512, 'reranker_timeout_seconds':5,
        'bm25_top_k':20, 'vector_top_k':40, 'rrf_k':60, 'query_rewrite':False, 'contextual_index':False,
        'policies':POLICIES, 'order':ROUTES, 'regular_pool_budget':20, 'worker_subquery_pool_budget':40,
        'selection':'Reserve top lexical/dense prefixes, deduplicate by chunk ID, refill by RRF; retain RRF order before reranking.',
        'overlap':'A common chunk satisfies both route-prefix reservations; no duplicated rerank slots.',
        'short_route':'An exhausted route contributes all available candidates; refill remaining slots by RRF.',
        'context':{'limit':5,'max_chars':6000,'per_source':2},
        'limits':['Same 124 development cases, not a blind test; known-span Hit@5 is not exhaustive Recall or answer accuracy.',
                  'Fixed route order, one warm CPU observation per route/query, not an SLA.',
                  'Three allocation ratios declared before evaluation, no case-specific policy selection.',
                  'Sparse route Top20 can be shorter than its target, e.g. 30 lexical reservations in a Worker pool40.',
                  'Vector/reranker fallback fails this benchmark; fallback handling is tested separately.'],
        'environment':{k:importlib.metadata.version(k) for k in ['pymongo','pymilvus','milvus-lite','sentence-transformers','torch']}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    reranker = CrossEncoderReranker(prior['reranker'],timeout=5)
    reranker.model = load_cross_encoder(prior['reranker'],revision=prior['reranker_revision'],
                                       device='cpu',trust_remote_code=False,max_length=512)
    reranker('预热问题',[{'title':'测试','content':'这是预热段落。'}])
    provider = RetrievalEmbeddingProvider(MODEL,revision=protocol['embedding_revision'],device='cpu')
    provider._load();provider.encode_query('预热问题')
    report = {'protocol':protocol,'cases':[],'summary':{}}
    try:
        with tempfile.TemporaryDirectory(prefix='rag-candidate-quota-') as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            server=Server(str(Path(directory)/'quota.db'),address=f'127.0.0.1:{port}')
            assert server.init() and server.start()
            try:
                store=MongoMilvusStore(args.mongo_uri,database='rag_quota_eval_'+uuid.uuid4().hex,
                    milvus_uri=f'http://127.0.0.1:{port}',provider=provider)
                try:
                    refs,cutoffs={},{}
                    for doc in docs:
                        item=store.ingest(doc['content'],uri=doc['uri'],title=doc['title'])
                        refs[item['snapshot_id']]=doc['id']
                        cutoffs[doc['id']]=store.db.versions.find_one({'_id':item['snapshot_id']})['published_at'].isoformat()+'+00:00'
                    lookup={c['_id']:{'document':refs[c['snapshot']],'start':c['start'],'end':c['end']} for c in store.db.chunks.find({})}
                    def format_ids(ids):return [lookup[cid] for cid in ids]
                    def format_hits(hits):
                        return [{'document':refs[h['snapshot_id']],'start':h['start'],'end':h['end'],'content':h['content']} for h in hits]
                    recording=RecordedReranker(reranker,format_hits)
                    store.reranker=recording
                    for ci,case in enumerate(cases):
                        row={k:case[k] for k in ['id','query','category','group']};row['routes']={}
                        for policy in ROUTES:
                            view=PolicyView(store,policy,format_ids)
                            recording.calls=[]
                            started=time.perf_counter()
                            if case['group']=='multi_source':
                                pack,_=build_context(view,case['subqueries'],limit=5,max_chars=6000,per_source=2)
                                raw=pack['passages']
                            else:
                                kwargs={'as_of':cutoffs[case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                                raw=view.search(case['query'],limit=5,mode='hybrid',**kwargs)
                            elapsed=(time.perf_counter()-started)*1000
                            hits=format_hits(raw)
                            if not recording.calls or any(h['rerank_status']!='applied' for h in raw):
                                raise RuntimeError(f'rerank fallback {case["id"]}/{policy}')
                            if policy=='rrf':
                                old=prior['models'][MODEL]['cases'][ci]
                                assert old['id']==case['id'] and old['routes']['on']['hits']==hits, 'baseline drift'
                                assert [(c['query'],c['input']) for c in recording.calls]==[(c['query'],c['input']) for c in old['rerank_calls']], 'baseline candidate drift'
                            else:
                                baseline=row['routes']['rrf']['channels']
                                assert len(view.channels)==len(baseline)
                                for current,base in zip(view.channels,baseline):
                                    assert all(current[k]==base[k] for k in ['query','budget','lexical','semantic','fused']), 'channel drift'
                            for call,channel in zip(recording.calls,view.channels):
                                assert call['count']==len(channel['selected'])<=channel['budget']
                                assert channel['selected']==channel['validated'], 'source gate drift in frozen corpus'
                                if policy!='rrf':
                                    reserved=channel['lexical'][:channel['lexical_quota']]+channel['semantic'][:channel['semantic_quota']]
                                    assert all(h in channel['selected'] for h in reserved), 'reservation not honored'
                            pool=[h for call in recording.calls for h in call['input']]
                            row['routes'][policy]={**score(case['evidence_groups'],hits),'hits':hits,'ms':elapsed,
                                'candidate_availability':score(case['evidence_groups'],pool,k=len(pool)),
                                'calls':list(recording.calls),'channels':view.channels}
                        report['cases'].append(row)
                        if (ci+1)%12==0 or ci+1==len(cases):
                            (args.output/'partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                            print(f'candidate quota paired {ci+1}/{len(cases)}',flush=True)
                    groups=defaultdict(list)
                    for row in report['cases']:groups[row['group']].append(row)
                    report['summary']={g:summarize(rows) for g,rows in groups.items()}
                    (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                    (args.output/'partial.json').unlink(missing_ok=True)
                    print(json.dumps(report['summary'],ensure_ascii=False),flush=True)
                finally:
                    store.mongo.drop_database(store.db.name);store.close()
            finally:server.stop()
    finally:reranker.executor.shutdown(wait=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mongo-uri',default='mongodb://127.0.0.1:27028')
    parser.add_argument('--output',type=Path,default=OUT)
    run(parser.parse_args())
