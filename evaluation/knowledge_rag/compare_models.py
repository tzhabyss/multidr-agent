"""Model-only recall comparison on actual MongoDB + Milvus, fixed v3 spans.

No dataset labels, chunk policy, candidate counts or RRF weights are tuned here.
Uses a disposable Mongo database and a private, real Milvus Lite server by default.
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

OUT=Path(__file__).parent/'model_comparison'


def summarize(rows):
    result={}
    for route in ['bm25','vector','hybrid']:
        values=[r['routes'][route] for r in rows]
        metric_keys=[k for k in values[0] if k not in ['hits','ms','context_chars']]
        result[route]={k:mean(v[k] for v in values) for k in metric_keys}
        result[route].update(count=len(values),p50_ms=percentile([v['ms'] for v in values],.5),
                             p95_ms=percentile([v['ms'] for v in values],.95))
    return result


def run(args):
    from huggingface_hub import snapshot_download
    from pymongo import MongoClient
    manifest,docs,cases=load_dataset()
    spans={d['id']:structured_chunks(d['content']) for d in docs}
    chunk_hash=hashlib.sha256(json.dumps(spans,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    results={'dataset_manifest':manifest,'chunk_policy':'structured-v1 size600; same exact spans for all models',
             'chunk_sha256':chunk_hash,'chunk_count_all_versions':sum(map(len,spans.values())),
             'candidate_counts':{'bm25':20,'vector':40,'rrf_k':60,'report_k':5},
             'reranker':None,'contextual_index':False,'models':{},
             'environment':{k:importlib.metadata.version(k) for k in ['pymongo','pymilvus','milvus-lite','sentence-transformers','torch']},
             'limits':['Known positive spans, not exhaustive chunk relevance: Hit@5 is not Recall@5.',
                       '124 development cases: 72 repository,32 synthetic,8 multi-source,12 no-answer; no new blind test set.',
                       'No LLM answer, hallucination or refusal accuracy is measured.',
                       'Single local run per query/route; p95 excludes model loading, not a service SLA.',
                       'Milvus Lite exact FLAT search; not Standalone ANN throughput or distribution testing.',
                       'Identical character chunks may exceed model-specific token budgets; counts are recorded, not hidden.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps({k:v for k,v in results.items() if k!='models'},ensure_ascii=False,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='recall-models-') as directory:
        from milvus_lite.server import Server
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        server=Server(str(Path(directory)/'models.db'),address=f'127.0.0.1:{port}')
        assert server.init() and server.start()
        try:
            for model_name in args.models:
                revision=Path(snapshot_download(model_name,local_files_only=True,
                    allow_patterns=['*.json','*.txt','*.safetensors','*.model'],
                    ignore_patterns=['onnx/*','openvino/*'])).name
                provider=RetrievalEmbeddingProvider(model_name,revision=revision,device=args.device)
                started=time.perf_counter();model=provider._load();load_seconds=time.perf_counter()-started
                provider.encode_query('预热查询')
                dbname='rag_model_eval_'+uuid.uuid4().hex
                store=MongoMilvusStore(args.mongo_uri,database=dbname,milvus_uri=f'http://127.0.0.1:{port}',provider=provider)
                try:
                    refs,cutoffs={},{}
                    started=time.perf_counter()
                    for doc in docs:
                        item=store.ingest(doc['content'],uri=doc['uri'],title=doc['title'])
                        refs[item['snapshot_id']]=doc['id']
                        cutoffs[doc['id']]=store.db.versions.find_one({'_id':item['snapshot_id']})['published_at'].isoformat()+'+00:00'
                    index_seconds=time.perf_counter()-started
                    texts=[d['content'][p['start']:p['end']] for d in docs for p in spans[d['id']]]
                    # Actual indexed strings include repeated split-table headers.
                    indexed=[store._index_text(v,r) for v in store.db.versions.find({})
                             for r in store.db.chunks.find({'snapshot':v['_id']})]
                    lengths=[len(model.tokenizer(provider.document_prefix+t,truncation=False)['input_ids']) for t in indexed]
                    rows=[]
                    for ci,case in enumerate(cases,1):
                        row={k:case[k] for k in ['id','group','category','query']};row['routes']={}
                        for route in ['bm25','vector','hybrid']:
                            started=time.perf_counter()
                            if case['group']=='multi_source':
                                pack,_=build_context(store,case['subqueries'],limit=5,max_chars=6000,per_source=2,search_mode=route)
                                hits=[{'document':refs[p['snapshot_id']],'start':p['start'],'end':p['end'],'content':p['content']}
                                      for p in pack['passages']]
                            else:
                                kwargs={'as_of':cutoffs[case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                                raw=store.search(case['query'],mode=route,limit=5,**kwargs)
                                if route!='bm25' and any(h['vector_status']!='available' for h in raw):
                                    raise RuntimeError('semantic fallback invalidates model comparison')
                                hits=[{'document':refs[h['snapshot_id']],'start':h['start'],'end':h['end'],'content':h['content']}
                                      for h in raw]
                            row['routes'][route]={**score(case['evidence_groups'],hits),'ms':(time.perf_counter()-started)*1000,'hits':hits}
                        rows.append(row)
                        if ci%24==0 or ci==len(cases):print(f'{model_name}: {ci}/{len(cases)}',flush=True)
                    groups=defaultdict(list);categories=defaultdict(list)
                    for row in rows:
                        groups[row['group']].append(row)
                        if row['group'] not in ['negative','multi_source']:
                            categories[row['group']+'/'+row['category']].append(row)
                    entry={'revision':revision,'model_id':provider.model_id,'degraded':provider.degraded,
                           'dimensions':provider.dimensions,'max_tokens':model.max_seq_length,
                           'token_overflow_chunks':sum(n>model.max_seq_length for n in lengths),'max_chunk_tokens':max(lengths),
                           'prefixes':MODEL_PRESETS[model_name],'device':str(model.device),
                           'parameter_count':sum(p.numel() for p in model.parameters()),
                           'load_seconds':load_seconds,'index_seconds':index_seconds,
                           'mongo_version':store.mongo.server_info()['version'],
                           'summary':{g:summarize(rs) for g,rs in groups.items()},
                           'by_category':{g:summarize(rs) for g,rs in categories.items()},'cases':rows}
                    # Keyword baseline MUST be identical across all embedding models.
                    if results['models']:
                        baseline=next(iter(results['models'].values()))
                        for old,new in zip(baseline['cases'],rows,strict=True):
                            if old['routes']['bm25']['hits']!=new['routes']['bm25']['hits']:
                                raise ValueError(f'BM25 baseline drift: {new["id"]}')
                    results['models'][model_name]=entry
                    (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
                    print(json.dumps({'model':model_name,'summary':entry['summary']},ensure_ascii=False),flush=True)
                finally:
                    store.mongo.drop_database(dbname);store.close()
                del store,provider,model;gc.collect()
        finally:
            server.stop()
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mongo-uri',default='mongodb://127.0.0.1:27028')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--models',nargs='+',choices=list(MODEL_PRESETS),default=list(MODEL_PRESETS))
    run(parser.parse_args())
