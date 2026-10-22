import asyncio
import json

import pytest

from deepresearch.knowledge.query_rewrite import RewrittenQuerySearch, rewrite_query, validate_rewrite
from deepresearch.llm import LLMResponse


def test_identifiers_dates_and_literal_query_terms_are_preserved():
    original = '2025年 MCP 用 run_id 恢复时，不使用 `--fresh` 怎么处理？'
    result = validate_rewrite(original, 'MCP 恢复研究的命令，不新建运行')
    assert all(term in result for term in ['2025年', 'MCP', 'run_id', '--fresh'])
    assert validate_rewrite(original, {'query':'no'}) is None
    assert validate_rewrite(original, 'a'*301) is None
    assert validate_rewrite(original, '') is None
    assert '--runtime-config' in validate_rewrite('如何使用 --runtime-config？', '怎样指定配置文件')


@pytest.mark.asyncio
async def test_llm_receives_only_question_and_records_actual_usage():
    class LLM:
        model_id = 'fake'
        async def complete_json(self, system, user, **kwargs):
            assert json.loads(user) == {'question':'MCP 凭证缺失怎么办？'}
            assert kwargs['max_tokens']==256
            return {'query':'MCP 缺少凭证时的处理行为'}, LLMResponse('', input_tokens=20, output_tokens=10)
    result = await rewrite_query(LLM(), 'MCP 凭证缺失怎么办？')
    assert result['status']=='rewritten'
    assert result['input_tokens']==20 and result['output_tokens']==10


@pytest.mark.asyncio
async def test_timeout_invalid_and_unchanged_rewrites_keep_original():
    class LLM:
        model_id = 'fake'
        async def complete_json(self, *args, **kwargs):
            await asyncio.sleep(1)
    assert (await rewrite_query(LLM(), '问题', timeout=.001))['status']=='fallback'
    class Invalid(LLM):
        async def complete_json(self, *args, **kwargs):
            return {'query':[]}, LLMResponse('')
    assert (await rewrite_query(Invalid(), '问题'))['status']=='invalid'
    class Unchanged(LLM):
        async def complete_json(self, *args, **kwargs):
            return {'query':'问题'}, LLMResponse('')
    assert (await rewrite_query(Unchanged(), '问题'))['status']=='unchanged'


def hit(cid, start=0):
    return {'chunk_id':cid, 'snapshot_id':'s', 'uri':'file:///a', 'start':start, 'end':start+10,
            'content':cid, 'title':'a', 'heading':[], 'rerank_status':'disabled'}


class Store:
    def __init__(self):
        self.queries = []
        self.reranker = lambda q, hs: hs
        self.original = [hit(f'a{i}', i*10) for i in range(60)]
        self.alternate = [hit(f'b{i}', 1000+i*10) for i in range(60)]
    def search(self, query, *, limit, rerank=True, **kwargs):
        self.queries.append((query, limit, rerank, kwargs))
        if query=='broken': raise RuntimeError('alternate unavailable')
        values = self.original if query=='original' else self.alternate
        return [dict(h) for h in values[:limit]]


def test_fixed_and_expanded_candidate_budgets_and_original_anchor():
    store = Store()
    identity = store.reranker
    rrf = RewrittenQuerySearch(store, {'original':'rewrite'}, record=True)
    rrf.search('original', limit=5, as_of='2025-01-01T00:00:00Z')
    assert len(rrf.calls[0]['input'])==20
    assert store.reranker is identity
    assert all(not q[2] and q[3]['as_of']=='2025-01-01T00:00:00Z' for q in store.queries)
    union = RewrittenQuerySearch(store, {'original':'rewrite'}, fusion='anchored_union', record=True)
    union.search('original', limit=5)
    assert len(union.calls[0]['input'])==40
    assert [h['chunk_id'] for h in union.calls[0]['input'][:20]]==[h['chunk_id'] for h in store.original[:20]]
    expanded = RewrittenQuerySearch(store, {}, multiplier=2, record=True)
    expanded.search('original', limit=5)
    assert len(expanded.calls[0]['input'])==40


def test_fused_reranker_uses_original_query_and_cannot_replace_source_bytes():
    store = Store()
    def rerank(query, hits):
        assert query=='original'
        for h in hits: h['content']='invented'
        return list(reversed(hits))
    store.reranker = rerank
    results = RewrittenQuerySearch(store, {'original':'rewrite'}).search('original', limit=5)
    assert all(h['content']==h['chunk_id'] and h['rerank_status']=='applied' for h in results)


def test_alternate_retrieval_error_retains_original_and_records_failure():
    store = Store()
    view = RewrittenQuerySearch(store, {'original':'broken'}, record=True)
    results = view.search('original', limit=5)
    assert [h['chunk_id'] for h in results]==[h['chunk_id'] for h in store.original[:5]]
    assert view.calls[0]['rewrite_error']=='RuntimeError'
    assert all(h['query_rewrite_status']=='fallback' for h in results)


def test_bad_reranker_falls_back_without_losing_or_accepting_unknown_candidates():
    store = Store()
    store.reranker = lambda q, hs: [hit('invented')]
    view = RewrittenQuerySearch(store, {'original':'rewrite'}, fusion='anchored_union')
    results = view.search('original', limit=5)
    assert all(h['rerank_status']=='fallback' and h['chunk_id'].startswith('a') for h in results)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_worker_optional_rewrite_falls_back_and_keeps_context_budget(tmp_path, failure):
    from deepresearch.agents import ResearchWorker
    from deepresearch.knowledge import KnowledgeStore
    from deepresearch.memory_system.embeddings import HashEmbeddingProvider
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.schemas import ResearchTask
    class LLM:
        model_id = 'test'
        async def complete_json(self, system, user, **kwargs):
            assert set(json.loads(user))=={'question'}
            if failure: raise RuntimeError('offline')
            return {'query':'报告费用金额'}, LLMResponse('', input_tokens=10, output_tokens=5)
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.backend = 'sqlite'
    config.knowledge.path = str(tmp_path/'library.sqlite')
    config.knowledge.query_rewrite_enabled = True
    config.knowledge.precontext_max_chars = 2000
    worker = ResearchWorker(LLM(), None, config)
    worker.knowledge = KnowledgeStore(config.knowledge.path, provider=HashEmbeddingProvider())
    worker.knowledge.ingest('报告费用为100元。', uri='file:///report.md', title='费用报告')
    task = ResearchTask.model_construct(task_id='t', objective='费用多少', key_questions=['费用多少'], queries=['费用多少'])
    pack, candidates, documents = await worker._knowledge_precontext(task)
    assert pack['passages'] and candidates and documents
    assert len(json.dumps(pack, ensure_ascii=False))<=2000
    assert pack['query_rewrite']['rewritten_queries']==(0 if failure else 1)
    assert pack['query_rewrite']['input_tokens']==(0 if failure else 10)
