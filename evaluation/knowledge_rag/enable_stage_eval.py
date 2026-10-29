"""Fixed-parameter RAG stage ablation on isolated Milvus namespaces.

Six routes share the frozen 124 development cases, GTE revision and budgets.
No live-library synthetic documents and no new LLM generations.
"""
from __future__ import annotations
import argparse,json,time,uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean
from deepresearch.runtime_config import load_runtime_config
from deepresearch.knowledge import create_store
from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.reranker import CrossEncoderReranker
from .run_v3 import load_dataset,percentile
from .metrics_v3 import score

STAGES=[('bm25','plain','bm25','rrf',False),('vector','plain','vector','rrf',False),
        ('hybrid_rrf','plain','hybrid','rrf',False),('quota_dense','plain','hybrid','quota_dense',False),
        ('background','background','hybrid','quota_dense',False),
        ('cross_encoder','background','hybrid','quota_dense',True)]


def run(output):
    manifest,docs,cases=load_dataset()
    active=load_runtime_config().knowledge
    # Application startup can fall back without this optional local artifact;
    # an ablation must not silently label a plain index as background-enhanced.
    if not active.document_context_path or not Path(active.document_context_path).is_file():
        raise ValueError('Stage ablation requires a generated document_context_path cache')
    ranker=CrossEncoderReranker.from_config(active)
    output.mkdir(parents=True,exist_ok=True)
    report={'protocol':{'dataset':manifest,'stages':[s[0] for s in STAGES],
        'embedding_model':active.embedding_model,'embedding_revision':active.embedding_revision,
        'reranker_model':active.reranker_model,'reranker_revision':active.reranker_revision,
        'device':active.embedding_device,'reranker_max_length':active.reranker_max_length,
        'bm25_top':20,'vector_top':40,'rrf_k':60,'regular_pool':20,'worker_pool':40,
        'quota_dense':'one quarter lexical, three quarters dense; overlap deduplicated',
        'context_chars':6000,'per_source':2,'query_rewrite':False,'source_neighbors':0,
        'limits':['Existing development cases; no held-out claim or answer accuracy.',
                  'Pool coverage uses known evidence groups, not exhaustive relevance Recall.',
                  'CPU single-run latency; no production SLA or cross-machine concurrency benchmark.',
                  'Runtime Worker Top3 differs from evaluation Top5.']},'cases':[],'summary':{}}
    stores={};refs={};cutoffs={};lookup={}
    try:
        provider=None
        for name in ['plain','background']:
            config=active.model_copy(update={'milvus_namespace':'stage_'+name+'_'+uuid.uuid4().hex,
                'document_context_path':None if name=='plain' else active.document_context_path})
            store=create_store(config,provider=provider,reranker=ranker)
            stores[name]=store;provider=store.provider;refs[name]={};cutoffs[name]={}
            for doc in docs:
                item=store.ingest(doc['content'],uri=doc['uri'],title=doc['title'])
                refs[name][item['snapshot_id']]=doc['id']
                cutoffs[name][doc['id']]=store.snapshot(item['snapshot_id'],collection='library')['published_at']
            versions=store.records.list('version')
            lookup[name]={c['_id']:{'document':refs[name][v['_id']], 'start':c['start'],'end':c['end'],
                'content':v['content'][c['start']:c['end']]} for v in versions for c in v['chunks']}
            assert len(lookup[name])==222
            print('indexed',name,'222 chunks',flush=True)
        ranker('预热',[{'title':'预热','content':'这是预热材料。'}])
        def format_hits(name,hits):
            values=[]
            for hit in hits:
                value=lookup[name][hit['chunk_id']]
                assert value['content']==hit['content']
                assert not hit.get('degraded',False)
                values.append(dict(value))
            return values
        prior=json.loads(Path('evaluation/knowledge_rag/context_rerank/results.json').read_text())
        for i,case in enumerate(cases):
            row={k:case[k] for k in ['id','query','group','category']};row['stages']={}
            for stage,name,mode,policy,rerank in STAGES:
                store=stores[name];traces=[]
                class View:
                    def snapshot(self,*a,**kw):return store.snapshot(*a,**kw)
                    def search(self,query,**kw):
                        kw.pop('mode',None);trace={}
                        hits=store.search(query,**kw,mode=mode,rerank=rerank,candidate_policy=policy,trace=trace)
                        if mode!='bm25':assert trace.get('vector_status')=='available'
                        if rerank:assert hits and all(h['rerank_status']=='applied' for h in hits)
                        traces.append({'query':query,**trace})
                        return hits
                started=time.perf_counter();before=ranker.completed_calls
                if case['group']=='multi_source':
                    pack,_=build_context(View(),case['subqueries'],limit=5,max_chars=6000,per_source=2)
                    raw=pack['passages']
                else:
                    kw={'as_of':cutoffs[name][case['as_of_snapshot']]} if case.get('as_of_snapshot') else {}
                    raw=View().search(case['query'],limit=5,**kw)
                elapsed=(time.perf_counter()-started)*1000
                hits=format_hits(name,raw)
                pool=list({cid:lookup[name][cid] for trace in traces for cid in trace.get('selected',[])}.values())
                group_score=score(case['evidence_groups'],pool,k=len(pool))
                row['stages'][stage]={**score(case['evidence_groups'],hits),'hits':hits,'ms':elapsed,
                    'pool':group_score,'pool_size':len(pool),'reranker_calls':ranker.completed_calls-before,
                    'traces':[{**t,**{k:[{key:v for key,v in lookup[name][cid].items() if key!='content'} for cid in t.get(k,[])]
                        for k in ['lexical','semantic','fused','selected','validated']}} for t in traces]}
            expected=prior['cases'][i]
            assert expected['id']==case['id']
            row['same_as_historical']=row['stages']['cross_encoder']['hits']==expected['routes']['bge_original']['hits']
            report['cases'].append(row)
            if (i+1)%12==0 or i+1==len(cases):
                (output/'partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                print('stage evaluation',i+1,'/',len(cases),flush=True)
        grouped=defaultdict(list)
        for row in report['cases']:grouped[row['group']].append(row)
        for group,rows in grouped.items():
            report['summary'][group]={}
            for stage,*_ in STAGES:
                values=[r['stages'][stage] for r in rows]
                keys=['candidate_nonempty'] if group=='negative' else ['hit_at_5','mrr_at_5','evidence_group_coverage_at_5','all_groups_at_5']
                summary={k:mean(v[k] for v in values) for k in keys}
                summary.update(count=len(rows),p50_ms=percentile([v['ms'] for v in values],.5),p95_ms=percentile([v['ms'] for v in values],.95),
                    mean_pool_size=mean(v['pool_size'] for v in values),reranker_calls=sum(v['reranker_calls'] for v in values))
                if group!='negative':summary['pool_known_group_coverage']=mean(v['pool']['evidence_group_coverage_at_5'] for v in values)
                report['summary'][group][stage]=summary
        report['historical_differences']=[r['id'] for r in report['cases'] if not r['same_as_historical']]
        (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        (output/'partial.json').unlink(missing_ok=True)
        print(json.dumps(report['summary']['repository'],ensure_ascii=False,indent=2),flush=True)
        print('historical differences',report['historical_differences'],flush=True)
    finally:
        # Isolated evaluation namespaces only; never drop the live library.
        for store in stores.values():
            names={j.get('vector_collection') for j in store.records.list('job')}-{None}
            for name in names:
                if store.vectors.has_collection(collection_name=name):store.vectors.drop_collection(collection_name=name)
            store.vectors.drop_collection(collection_name=store.records.name)
        if stores:next(iter(stores.values())).close()
        ranker.executor.shutdown(wait=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('evaluation/knowledge_rag/enabled_stages'))
    run(parser.parse_args().output)
