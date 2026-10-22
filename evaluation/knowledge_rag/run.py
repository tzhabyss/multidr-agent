"""Small manually labelled repository-document retrieval benchmark (development only).

No LLM calls. Corpus is unmodified checked-in documentation; labels are literal
source spans. This is not a held-out benchmark or report-quality evaluation.
"""
from __future__ import annotations
import argparse
import json
import tempfile
import time
from pathlib import Path
from statistics import mean

from deepresearch.knowledge import KnowledgeStore
from deepresearch.memory_system.embeddings import HashEmbeddingProvider, ResilientEmbeddingProvider
from deepresearch.agents import _focused_extraction_windows
from deepresearch.schemas import ResearchTask

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ('README.md', '系统证据如何逐级追溯到网页快照？', 'Source -> SourceSnapshot -> Evidence -> Claim -> ClaimEvidenceLink -> Citation'),
    ('README.md', '普通研究默认会同时请求哪两个搜索服务？', 'deepseek-web-search` 与 `anysearch-search'),
    ('README.md', '一轮研究发现资料不足时怎么办？', 'need_more -> 结构化缺口返回 Planner'),
    ('README.md', '如何更换整个项目的运行配置文件？', 'DEEPRESEARCH_CONFIG'),
    ('docs/model-modes.md', 'Which mode has eight workers and six rounds?', '| `max` | high-value, difficult research | 8 | 6 |'),
    ('docs/model-modes.md', 'How can I change mode within an interactive session?', '/mode deep'),
    ('docs/model-modes.md', 'Does choosing a stronger model grant additional tool permissions?', 'Tool access is intentionally not granted by model tier.'),
    ('docs/model-modes.md', 'What does model_tiers configure?', 'maps the logical `lite`, `default`, and `craft` tiers'),
    ('docs/mcp.md', '没有 SERPAPI_KEY 时学术搜索先用什么服务？', 'Semantic Scholar first when `SERPAPI_KEY` is absent'),
    ('docs/mcp.md', '如何安装本地PDF和Word支持？', 'pip install -e ".[documents]"'),
    ('docs/mcp.md', 'Is an OpenAlex project key mandatory?', 'OpenAlex currently works through its public API without a project API key.'),
    ('docs/mcp.md', '外部工具如何限制角色和权限？', 'only explicitly allowlisted MCP tools'),
]


def run(output, lexical=False):
    corpus = {name: (ROOT/name).read_text() for name, _, _ in CASES}
    for name, _, gold in CASES:
        assert gold in corpus[name], (name, gold)
    provider = HashEmbeddingProvider() if lexical else ResilientEmbeddingProvider(fallback_enabled=False)
    with tempfile.TemporaryDirectory() as directory:
        store = KnowledgeStore(Path(directory)/'bench.sqlite', provider=provider)
        start = time.perf_counter()
        ids = {name: store.ingest(text, uri=(ROOT/name).as_uri(), title=name)['snapshot_id'] for name,text in corpus.items()}
        index_seconds = time.perf_counter()-start
        results = []
        for name, query, gold in CASES:
            text = corpus[name]
            a, b = text.index(gold), text.index(gold)+len(gold)
            task = ResearchTask.model_construct(research_object=query, research_dimension='', target_entity_ids=[], target_field_keys=[], key_questions=[query])
            start = time.perf_counter()
            baseline, _ = _focused_extraction_windows(text, task)
            # Actual current extraction also falls back to the initial content window.
            if not baseline:
                from deepresearch.agents import _content_windows, _merge_extraction_windows, WORKER_MAX_CHUNKS_PER_SOURCE
                merged = _merge_extraction_windows(_content_windows(text)[:WORKER_MAX_CHUNKS_PER_SOURCE])
                baseline = [merged] if merged else []
            baseline_ms = (time.perf_counter()-start)*1000
            item = {'query':query,'source':name,'keyword_window':{'span_recall':int(any(x<=a and y>=b for x,y,*_ in baseline)), 'context_chars':sum(len(w[2]) for w in baseline),'ms':baseline_ms}}
            for mode in ('bm25','vector','hybrid'):
                if lexical and mode=='vector':
                    continue
                start=time.perf_counter()
                hits=store.search(query, limit=5, mode=mode)
                ranks=[i for i,h in enumerate(hits,1) if h['snapshot_id']==ids[name] and h['start']<=a and h['end']>=b]
                windows=store.windows(query,snapshot_id=ids[name],collection='library',limit=3) if mode=='hybrid' else []
                item[mode]={'recall_at_5':int(bool(ranks)), 'reciprocal_rank':1/min(ranks) if ranks else 0,
                            'ms':(time.perf_counter()-start)*1000,'degraded':provider.degraded}
                if mode=='hybrid':
                    item[mode].update(span_recall=int(any(x<=a and y>=b for x,y,*_ in windows)),context_chars=sum(len(w[2]) for w in windows))
            results.append(item)
        summary={}
        for mode in ('keyword_window','bm25','vector','hybrid'):
            rows=[r[mode] for r in results if mode in r]
            if rows:
                summary[mode]={key:mean(r[key] for r in rows) for key in rows[0] if key!='degraded'}
        report={'dataset':'12 manually labelled bilingual questions on 3 repository documents; development set',
                'model':provider.model_id,'degraded':provider.degraded,'index_seconds':index_seconds,
                'limitations':['No held-out set, no LLM extraction or final report scoring.',
                    'Character count is a context-size proxy, not measured API tokens.',
                    'Keyword window and hybrid span_recall use the known source; Recall@5/MRR search the entire corpus.',
                    'Hybrid latency includes passage-window retrieval; BM25/vector latency is search only.',
                    'No reranker in this run. Captured-time history is not document effective-date governance.'],
                'summary':summary,'cases':results}
        Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='evaluation/knowledge_rag/results.json')
    parser.add_argument('--lexical-only',action='store_true')
    args=parser.parse_args()
    run(args.output,args.lexical_only)
