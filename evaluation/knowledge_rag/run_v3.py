"""Frozen 124-case RAG evaluation. Does not modify retrieval parameters or labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from deepresearch.knowledge import KnowledgeStore
from deepresearch.knowledge.context import build_context
from deepresearch.knowledge.reranker import CrossEncoderReranker
from deepresearch.memory_system.embeddings import ResilientEmbeddingProvider
from .metrics_v3 import score

DATA=Path(__file__).parent/'v3'


def load_dataset(directory=DATA):
    manifest=json.loads((directory/'manifest.json').read_text())
    for file,key in [('corpus.json','corpus_sha256'),('cases.jsonl','cases_sha256')]:
        if hashlib.sha256((directory/file).read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError(f'frozen dataset hash mismatch: {file}')
    docs=json.loads((directory/'corpus.json').read_text())
    cases=[json.loads(line) for line in (directory/'cases.jsonl').read_text().splitlines() if line]
    if len(cases)!=manifest['case_count'] or len(docs)!=manifest['document_count']:
        raise ValueError('manifest counts differ')
    ids={d['id']:d for d in docs}
    if len(ids)!=len(docs) or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('duplicate document/case IDs')
    if len({c['query'] for c in cases})!=len(cases):
        raise ValueError('duplicate questions')
    for case in cases:
        for group in case['evidence_groups']:
            if not group:raise ValueError('empty positive evidence group')
            for alt in group:
                text=ids[alt['document']]['content']
                if text[alt['start']:alt['end']]!=alt['text']:
                    raise ValueError(f'invalid label: {case["id"]}')
    return manifest,docs,cases


def percentile(values,p):
    values=sorted(values)
    return values[min(len(values)-1,round((len(values)-1)*p))]


def summarize(rows):
    result={}
    for stage in rows[0]['stages']:
        values=[row['stages'][stage] for row in rows]
        metrics=[k for k in values[0] if k not in {'hits','pack','ms','context_chars'}]
        result[stage]={k:mean(v[k] for v in values) for k in metrics}
        result[stage].update(mean_ms=mean(v['ms'] for v in values),
                             p50_ms=percentile([v['ms'] for v in values],.5),
                             p95_ms=percentile([v['ms'] for v in values],.95))
        if 'context_chars' in values[0]:
            result[stage]['mean_context_chars']=mean(v['context_chars'] for v in values)
    return result


def run(args):
    manifest,docs,cases=load_dataset()
    provider=ResilientEmbeddingProvider(fallback_enabled=False)
    reranker=None
    if args.reranker_model:
        from sentence_transformers import CrossEncoder
        reranker=CrossEncoderReranker(args.reranker_model,timeout=30)
        reranker.model=CrossEncoder(args.reranker_model)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    start_all=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as directory:
            stores={name:KnowledgeStore(Path(directory)/f'{name}.sqlite',provider=provider,contextual=flag)
                    for name,flag in [('plain',False),('contextual',True)]}
            refs,cutoffs={},{}
            index_start=time.perf_counter()
            for name,store in stores.items():
                refs[name]={};cutoffs[name]={}
                for doc in docs:
                    item=store.ingest(doc['content'],uri=doc['uri'],title=doc['title'])
                    refs[name][item['snapshot_id']]=doc['id']
                    cutoffs[name][doc['id']]=store.snapshot(item['snapshot_id'],collection='library')['created_at']
            index_seconds=time.perf_counter()-index_start
            rows=[]
            for ci,case in enumerate(cases,1):
                row={'id':case['id'],'group':case['group'],'category':case['category'],
                     'query':case['query'],'stages':{}}
                if case['group']=='multi_source':
                    store=stores['plain']
                    store.reranker=reranker
                    for name,multi in [('single_query_context',False),('subquestion_context',True)]:
                        start=time.perf_counter()
                        pack,_=build_context(store,case['subqueries'],limit=5,max_chars=6000,per_source=2,multi_query=multi)
                        hits=[{'document':refs['plain'][p['snapshot_id']], 'start':p['start'],'end':p['end'],
                               'content':p['content'],'source_uri':p['source_uri']} for p in pack['passages']]
                        if reranker and any(p['rerank_status']!='applied' for p in pack['passages']):
                            raise RuntimeError('context reranker fell back')
                        row['stages'][name]={**score(case['evidence_groups'],hits),
                            'ms':(time.perf_counter()-start)*1000,
                            'context_chars':len(json.dumps(pack,ensure_ascii=False)), 'hits':hits}
                    store.reranker=None
                else:
                    for name,store in stores.items():
                        store.reranker=None
                        kwargs={}
                        if case.get('as_of_snapshot'):
                            kwargs['as_of']=cutoffs[name][case['as_of_snapshot']]
                        start=time.perf_counter()
                        pool=store.search(case['query'],limit=20,**kwargs)
                        elapsed=(time.perf_counter()-start)*1000
                        def format_hits(items):
                            return [{'document':refs[name][h['snapshot_id']], 'start':h['start'], 'end':h['end'],
                                     'content':h['content'],'heading':h['heading']} for h in items[:5]]
                        hits=format_hits(pool)
                        row['stages'][f'{name}_hybrid']={**score(case['evidence_groups'],hits),'ms':elapsed,'hits':hits}
                        if reranker:
                            rank_start=time.perf_counter()
                            ranked=reranker(case['query'],[dict(h) for h in pool]) if pool else []
                            if sorted(h['chunk_id'] for h in ranked)!=sorted(h['chunk_id'] for h in pool):
                                raise ValueError('invalid reranker permutation')
                            hits=format_hits(ranked)
                            row['stages'][f'{name}_rerank']={**score(case['evidence_groups'],hits),
                                'ms':elapsed+(time.perf_counter()-rank_start)*1000,'hits':hits}
                rows.append(row)
                if ci%12==0 or ci==len(cases):
                    print(f'evaluated {ci}/{len(cases)}',flush=True)
            groups=defaultdict(list)
            categories=defaultdict(list)
            for row in rows:
                groups[row['group']].append(row)
                if row['group'] not in {'negative','multi_source'}:
                    categories[f'{row["group"]}/{row["category"]}'].append(row)
            report={'dataset_manifest':manifest,'model':provider.model_id,'degraded':provider.degraded,
                    'reranker_model':args.reranker_model,'index_seconds':index_seconds,
                    'total_seconds':time.perf_counter()-start_all,
                    'summary':{g:summarize(rs) for g,rs in groups.items()},
                    'by_category':{g:{'count':len(rs),'metrics':summarize(rs)} for g,rs in categories.items()},
                    'metric_definitions':{
                        'hit_at_5':'At least one required evidence group has a known valid alternative covered by top5 excerpts.',
                        'mrr_at_5':'Reciprocal earliest prefix rank supporting a required evidence group.',
                        'evidence_group_coverage_at_5':'Fraction of required fact groups supported by top5 excerpts.',
                        'all_groups_at_5':'All required evidence groups supported.',
                        'candidate_nonempty':'For no-answer cases only: any candidate returned; not a hallucination or abstention metric.'},
                    'limits':['Known positive alternatives are not exhaustive semantic relevance annotations.',
                              'Development and synthetic diagnostics; not an independent hidden benchmark.',
                              'Question count is not independent scenario count; synthetic templates share structure.',
                              'No live LLM answer or Worker strategy quality judged; no production algorithm tuned in this expansion.',
                              'Ranks use a 20-candidate pool; mean/p95 latency excludes model loading and reflects this local run.'],
                    'cases':rows}
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k in {'model','reranker_model','summary','total_seconds'}},ensure_ascii=False,indent=2),flush=True)
    finally:
        if reranker:reranker.executor.shutdown(wait=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(DATA/'results.json'))
    parser.add_argument('--reranker-model',default='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    run(parser.parse_args())
