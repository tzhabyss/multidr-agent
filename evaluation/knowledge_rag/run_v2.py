"""Ablations for contextual indexing, real reranking and Worker precontext.

Repository questions and synthetic metadata diagnostics are reported separately.
No quality claims about final research answers are made by this retrieval test.
"""
from __future__ import annotations
import argparse
import json
import tempfile
import time
from pathlib import Path
from statistics import mean

from deepresearch.knowledge import KnowledgeStore
from deepresearch.knowledge.context import build_context
from deepresearch.memory_system.embeddings import ResilientEmbeddingProvider
from .run import ROOT, CASES


def diagnostic_corpus():
    docs, cases = {}, []
    for company, amounts in [('星河',(7.2,8.5)),('远山',(2.4,3.1)),('海川',(5.0,5.8)),('晨光',(1.1,1.6))]:
        for year, amount in zip((2023,2024),amounts):
            key = f'{company}-{year}.md'
            title = f'{company}公司{year}年度报告'
            text = f'# 经营概览\n\n公司持续发展主营业务，积极改善经营效率。\n\n## 研发投入\n\n研发投入为{amount}亿元。统计口径包含费用化与资本化研发支出，金额单位为人民币。\n\n## 风险提示\n\n本报告的数据仅对应本报告期，不代表未来经营结果。'
            docs[key]=(title,text)
            if year == 2024:
                cases.append((key,f'{company}公司2024年的研发投入是多少？',f'研发投入为{amount}亿元。'))
    for name, unit, answer in [('微光','百万次请求','100元'),('流星','百万输入Token','120元'),('云帆','百万输出Token','300元'),('青禾','每小时计算资源','8元')]:
        key=f'{name}-pricing.md'
        docs[key]=(f'{name}服务计费说明',f'# 价格\n\n标准档价格为{answer}，计价单位为{unit}。\n\n# 适用条件\n\n价格不含税费；不同计价单位的金额不能直接比较。')
        cases.append((key,f'{name}服务标准档按什么单位计费？',f'计价单位为{unit}'))
    return docs,cases


def summarize(rows):
    return {stage:{k:mean(r[stage][k] for r in rows) for k in ('recall_at_5','mrr_at_5','ms')}
            for stage in rows[0] if stage not in {'query','source','gold'} and isinstance(rows[0][stage],dict)}


def run(args):
    repository={name:(name,(ROOT/name).read_text()) for name,_,_ in CASES}
    synthetic, diagnostics=diagnostic_corpus()
    corpus={**repository,**synthetic}
    provider=ResilientEmbeddingProvider(fallback_enabled=False)
    reranker=None
    if args.reranker_model:
        from sentence_transformers import CrossEncoder
        from deepresearch.knowledge.reranker import CrossEncoderReranker
        reranker=CrossEncoderReranker(args.reranker_model,timeout=30)
        reranker.model=CrossEncoder(args.reranker_model)
    try:
        with tempfile.TemporaryDirectory() as directory:
            stores={name:KnowledgeStore(Path(directory)/f'{name}.sqlite',provider=provider,contextual=contextual)
                    for name,contextual in [('plain',False),('contextual',True)]}
            index_start=time.perf_counter()
            ids={name:{} for name in stores}
            for name,store in stores.items():
                for key,(title,text) in corpus.items():
                    ids[name][key]=store.ingest(text,uri=f'file:///benchmark/{key}',title=title)['snapshot_id']
            stages=[('plain_hybrid','plain',None),('contextual_hybrid','contextual',None)]
            if reranker:
                stages.extend([('plain_rerank','plain',reranker), ('contextual_rerank','contextual',reranker)])
            index_seconds=time.perf_counter()-index_start
            groups={}
            all_results={}
            for group,cases in [('repository_development',CASES),('synthetic_metadata_diagnostic',diagnostics)]:
                rows=[]
                for key,query,gold in cases:
                    text=corpus[key][1]
                    a,b=text.index(gold),text.index(gold)+len(gold)
                    row={'query':query,'source':key,'gold':gold}
                    for stage,name,ranker in stages:
                        store=stores[name]
                        store.reranker=ranker
                        start=time.perf_counter()
                        hits=store.search(query,limit=5)
                        ms=(time.perf_counter()-start)*1000
                        ranks=[i for i,h in enumerate(hits,1) if h['snapshot_id']==ids[name][key] and h['start']<=a and h['end']>=b]
                        if ranker and hits and any(h['rerank_status']!='applied' for h in hits):
                            raise RuntimeError('reranker fell back: do not report this as a reranking result')
                        row[stage]={'recall_at_5':int(bool(ranks)),'mrr_at_5':1/min(ranks) if ranks else 0,'ms':ms}
                    rows.append(row)
                groups[group]=summarize(rows)
                all_results[group]=rows
            # A known cross-source task, identical budget with and without multi-query.
            store=stores['contextual']
            store.reranker=reranker
            queries=['星河公司2024年的研发投入是多少？','海川公司2024年的研发投入是多少？','流星服务按什么单位计费？']
            expected=['星河-2024.md','海川-2024.md','流星-pricing.md']
            packs={}
            for name,multi in [('single_query',False),('task_subquestions',True)]:
                start=time.perf_counter()
                pack,_=build_context(store,queries,limit=5,max_chars=6000,per_source=2,multi_query=multi)
                found={p['source_uri'].rsplit('/',1)[-1] for p in pack['passages']}
                packs[name]={'expected_source_recall':sum(s in found for s in expected)/len(expected),
                             'source_count':len(found),'context_chars':len(json.dumps(pack,ensure_ascii=False)),
                             'ms':(time.perf_counter()-start)*1000,'pack':pack}
            report={'model':provider.model_id,'degraded':provider.degraded,'reranker_model':args.reranker_model,
                    'index_seconds':index_seconds,'corpus_documents':len(corpus),
                    'dataset_counts':{'repository_development':len(CASES),'synthetic_metadata_diagnostic':len(diagnostics)},
                    'summary':groups,'precontext':packs,'cases':all_results,
                    'limitations':['Development questions and hand-authored diagnostics, not a held-out benchmark.',
                                   'Repository questions now search a larger shared corpus; not directly comparable to v1 numbers.',
                                   'Metadata diagnostic intentionally requires document identity; reported separately to avoid overstating general gains.',
                                   'No live LLM strategy or final-answer accuracy evaluation; tests verify pipeline order and boundaries.',
                                   'Context chars are serialized characters, not measured model tokens.']}
            Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k not in {'cases','precontext'}},ensure_ascii=False,indent=2))
            print(json.dumps({k:{a:b for a,b in v.items() if a!='pack'} for k,v in packs.items()},ensure_ascii=False,indent=2))
    finally:
        if reranker:
            reranker.executor.shutdown(wait=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='evaluation/knowledge_rag/results-v2.json')
    parser.add_argument('--reranker-model')
    run(parser.parse_args())
