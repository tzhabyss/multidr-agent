"""Paired, fail-closed reranker ablation on the actual dual-store search path.

Regular Top5 searches rerank 20 fused candidates. Multi-source Worker context
searches rerank up to 40 per subquery, matching build_context/store.search.
"""
from __future__ import annotations
import argparse
import gc
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

from .run_v3 import load_dataset, percentile
from .metrics_v3 import score
from deepresearch.knowledge.dual_store import MongoMilvusStore, structured_chunks
from deepresearch.knowledge.embeddings import RetrievalEmbeddingProvider, MODEL_PRESETS
from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.reranker import CrossEncoderReranker, load_cross_encoder

OUT=Path(__file__).parent/'rerank_comparison'
MODEL_RESULTS=Path(__file__).parent/'model_comparison/results.json'


def summarize(rows):
    result={}
    for route in ['off','on']:
        values=[r['routes'][route] for r in rows]
        keys=[k for k in values[0] if k not in ['hits','ms','context_chars']]
        result[route]={k:mean(v[k] for v in values) for k in keys}
        result[route].update(count=len(values),p50_ms=percentile([v['ms'] for v in values],.5),
                             p95_ms=percentile([v['ms'] for v in values],.95))
    return result


class RecordedReranker:
    def __init__(self, reranker, format_hits):
        self.reranker, self.format_hits = reranker, format_hits
        self.calls=[]
    def __call__(self, query, hits):
        start=time.perf_counter()
        ranked=self.reranker(query,hits)
        self.calls.append({'query':query,'count':len(hits),'input':self.format_hits(hits),
                           'output':self.format_hits(ranked),'rerank_ms':(time.perf_counter()-start)*1000})
        return ranked


def run(args):
    from huggingface_hub import snapshot_download
    from milvus_lite.server import Server
    manifest,docs,cases=load_dataset()
    prior=json.loads(MODEL_RESULTS.read_text())
    reference=json.loads(args.reference.read_text()) if args.reference else None
    spans={d['id']:structured_chunks(d['content']) for d in docs}
    chunk_hash=hashlib.sha256(json.dumps(spans,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if prior['chunk_sha256']!=chunk_hash or prior['dataset_manifest']!=manifest:
        raise ValueError('frozen dataset/chunks changed')
    revision=Path(snapshot_download(args.reranker,local_files_only=True,
        allow_patterns=['*.json','*.txt','*.safetensors','*.model'],ignore_patterns=['onnx/*','openvino/*'])).name
    reranker=CrossEncoderReranker(args.reranker,timeout=30)
    started=time.perf_counter()
    reranker.model=load_cross_encoder(args.reranker,revision=revision,device='cpu',trust_remote_code=False,
                                    max_length=args.max_length)
    reranker_load=time.perf_counter()-started
    reranker('预热问题',[{'title':'测试','content':'这是预热段落。'}])
    report={'dataset_manifest':manifest,'chunk_sha256':chunk_hash,'reranker':args.reranker,
            'reranker_revision':revision,'reranker_load_seconds':reranker_load,'device':'cpu',
            'reranker_parameters':sum(p.numel() for p in reranker.model.model.parameters()),
            'candidate_reference':str(args.reference) if args.reference else None,
            'protocol':{'retrieval_mode':'hybrid','bm25':20,'vector':40,'rrf_k':60,'contextual_index':False,
                        'final_k':5,'regular_rerank_pool':20,'worker_subquery_rerank_pool':40,
                        'reranker_max_length':args.max_length},
            'environment':{k:importlib.metadata.version(k) for k in ['pymongo','pymilvus','milvus-lite','sentence-transformers','torch']},
            'limits':['Same 124 development cases, not an independent blind test.',
                      'Known positive evidence spans are not exhaustive relevance labels; Hit@5 is not Recall@5.',
                      'No answer generation, citation accuracy, hallucination or refusal rate is evaluated.',
                      'Warm local CPU timings, one observation per route per query; not an SLA.',
                      'Reranker unavailable or malformed results cause this experiment to fail, not silently fall back.',
                      'Candidate availability is a loose upper bound before source/character budgets.'], 'models':{}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps({k:v for k,v in report.items() if k!='models'},ensure_ascii=False,indent=2)+'\n')
    try:
        with tempfile.TemporaryDirectory(prefix='rerank-paired-') as directory:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            server=Server(str(Path(directory)/'rerank.db'),address=f'127.0.0.1:{port}')
            assert server.init() and server.start()
            try:
                for model_name in args.models:
                    old=prior['models'][model_name]
                    provider=RetrievalEmbeddingProvider(model_name,revision=old['revision'],device='cpu')
                    provider._load();provider.encode_query('预热问题')
                    store=MongoMilvusStore(args.mongo_uri,database='rag_rerank_eval_'+uuid.uuid4().hex,
                        milvus_uri=f'http://127.0.0.1:{port}',provider=provider)
                    try:
                        refs,cutoffs={},{}
                        for doc in docs:
                            item=store.ingest(doc['content'],uri=doc['uri'],title=doc['title'])
                            refs[item['snapshot_id']]=doc['id']
                            cutoffs[doc['id']]=store.db.versions.find_one({'_id':item['snapshot_id']})['published_at'].isoformat()+'+00:00'
                        def format_hits(hits):
                            return [{'document':refs[h['snapshot_id']],'start':h['start'],'end':h['end'],'content':h['content']} for h in hits]
                        recording=RecordedReranker(reranker,format_hits)
                        rows=[]
                        for ci,case in enumerate(cases,1):
                            row={k:case[k] for k in ['id','group','category','query']};row['routes']={}
                            old_case=old['cases'][ci-1]
                            for route in ['off','on']:
                                store.reranker=recording if route=='on' else None
                                recording.calls=[]
                                started=time.perf_counter()
                                if case['group']=='multi_source':
                                    pack,_=build_context(store,case['subqueries'],limit=5,max_chars=6000,per_source=2,search_mode='hybrid')
                                    raw=pack['passages']
                                    hits=format_hits(raw)
                                else:
                                    kwargs={'as_of':cutoffs[case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                                    raw=store.search(case['query'],limit=5,mode='hybrid',**kwargs)
                                    hits=format_hits(raw)
                                    if any(h['vector_status']!='available' for h in raw):
                                        raise RuntimeError('vector fallback invalidates paired comparison')
                                if route=='on' and (not recording.calls or any(h['rerank_status']!='applied' for h in raw)):
                                    raise RuntimeError(f'rerank fallback for {case["id"]}')
                                row['routes'][route]={**score(case['evidence_groups'],hits),
                                    'ms':(time.perf_counter()-started)*1000,'hits':hits}
                                if route=='off' and hits!=old_case['routes']['hybrid']['hits']:
                                    raise ValueError(f'baseline drift {model_name}/{case["id"]}')
                                if route=='on':
                                    row['rerank_calls']=recording.calls
                                    if reference:
                                        expected=reference['models'][model_name]['cases'][ci-1]
                                        actual_inputs=[(c['query'],c['input']) for c in recording.calls]
                                        expected_inputs=[(c['query'],c['input']) for c in expected['rerank_calls']]
                                        if expected['id']!=case['id'] or actual_inputs!=expected_inputs:
                                            raise ValueError(f'cross-model candidate drift {model_name}/{case["id"]}')
                                    if case['group']!='multi_source' and recording.calls[0]['input'][:5]!=row['routes']['off']['hits']:
                                        raise ValueError('on/off pre-rerank candidate drift')
                            if case['evidence_groups']:
                                pool=[h for call in recording.calls for h in call['input']]
                                upper=score(case['evidence_groups'],pool,k=len(pool))
                                row['candidate_availability']={'any_group':upper['hit_at_5'],
                                    'all_groups':upper['all_groups_at_5'],'group_coverage':upper['evidence_group_coverage_at_5']}
                            rows.append(row)
                            if ci%12==0 or ci==len(cases):print(f'{model_name}: paired {ci}/{len(cases)}',flush=True)
                        groups=defaultdict(list)
                        for row in rows:groups[row['group']].append(row)
                        entry={'embedding_revision':old['revision'],'dimensions':provider.dimensions,
                               'summary':{g:summarize(rs) for g,rs in groups.items()},'cases':rows}
                        report['models'][model_name]=entry
                        (args.output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                        print(json.dumps({'model':model_name,'summary':entry['summary']},ensure_ascii=False),flush=True)
                    finally:
                        store.mongo.drop_database(store.db.name);store.close()
                    del store,provider;gc.collect()
            finally:
                server.stop()
    finally:
        reranker.executor.shutdown(wait=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mongo-uri',default='mongodb://127.0.0.1:27028')
    parser.add_argument('--reranker',default='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    parser.add_argument('--models',nargs='+',choices=list(MODEL_PRESETS),default=['thenlper/gte-small-zh','BAAI/bge-small-zh-v1.5','intfloat/multilingual-e5-small'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--reference',type=Path,help='Require identical ordered candidate pools to this prior paired run')
    parser.add_argument('--max-length',type=int,default=512)
    run(parser.parse_args())
