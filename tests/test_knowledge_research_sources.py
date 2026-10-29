"""Cross-run original-source reuse; no paid model calls or semantic claims."""
import asyncio
import json

import pytest

from deepresearch.agents import ResearchWorker
from deepresearch.knowledge import KnowledgeStore
from deepresearch.knowledge.research_sources import capture_metadata, research_collection
from deepresearch.memory_system.embeddings import HashEmbeddingProvider
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import FetchedDocument, ResearchTask
from tests.fakes import FakeLLM


class CountingProvider(HashEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode_documents(self, values):
        self.calls += 1
        return super().encode_documents(values)


def worker(tmp_path, *, enabled=True, persist=True):
    config = load_runtime_config()
    config.knowledge.enabled = enabled
    config.knowledge.backend = 'sqlite'
    config.knowledge.path = str(tmp_path / 'sources.sqlite')
    config.knowledge.persist_fetched_sources = persist
    result = ResearchWorker(FakeLLM(), None, config)
    if enabled:
        result.knowledge = KnowledgeStore(config.knowledge.path, provider=CountingProvider())
    return result


def state(run='run-1', user='alice', project='project'):
    return {'run_id': run, 'user_id': user, 'project_id': project, 'request': {}}


def document(content='产品续航为10小时。', captured='2026-10-24T08:00:00+00:00', **updates):
    return FetchedDocument(requested_url='https://example.org/product',
        final_url='https://example.org/product', title='产品原始说明', content=content,
        fetched_at=captured, content_type='text/html', http_status=200).model_copy(update=updates)


def task():
    return ResearchTask.model_construct(task_id='t1', objective='产品续航',
        queries=['产品续航'], key_questions=['产品续航'], target_field_keys=[])


@pytest.mark.asyncio
async def test_cross_run_restart_reuse_original_metadata_and_no_duplicate_embedding(tmp_path):
    first = worker(tmp_path)
    doc = document()
    windows = await first._knowledge_windows(doc, task(), 'run-1', state=state(),
                                             published_at='2026-10-01')
    assert windows and windows[0][2] in doc.content
    second_doc = document(captured='2026-10-24T09:00:00+00:00')
    await asyncio.gather(*(first._knowledge_windows(second_doc, task(), 'run-2',
        state=state('run-2')) for _ in range(3)))
    assert first.knowledge.provider.calls == 1
    # Fresh worker/store instance has no in-memory linkage to the first run.
    reopened = worker(tmp_path)
    pack, candidates, docs = await reopened._knowledge_precontext(task(), state('run-3'))
    assert pack['status'] == 'retrieved'
    excerpt = pack['passages'][0]
    assert excerpt['content'] in doc.content
    assert excerpt['captured_at'] == doc.fetched_at
    assert excerpt['last_fetched_at'] == second_doc.fetched_at
    assert excerpt['published_at'] == '2026-10-01'
    assert excerpt['requires_online_verification'] is True
    assert excerpt['collection'] == research_collection(state())
    # Cached HTTP sources are dated leads, not materialized local evidence.
    assert candidates == [] and docs == {}
    assert not reopened.knowledge.search('续航', collection='library')
    assert not reopened.knowledge.search('续航', collection='run:run-1')
    snapshot = reopened.knowledge.snapshot(excerpt['snapshot_id'], collection=excerpt['collection'])
    meta = json.loads(snapshot['metadata'])
    assert meta['origin_run_id'] == 'run-1' and meta['last_seen_run_id'] == 'run-2'
    assert snapshot['content'] == doc.content


@pytest.mark.asyncio
async def test_scope_separation_missing_scope_and_opt_out(tmp_path):
    original = worker(tmp_path)
    await original._knowledge_windows(document(), task(), 'run-1', state=state())
    for different in (state(user='bob'), state(project='other'), {}, None):
        pack, _, _ = await original._knowledge_precontext(task(), different)
        assert pack['passages'] == []
    assert research_collection(state(user='a:b', project='c')) != research_collection(state(user='a', project='b:c'))
    opted_out = worker(tmp_path, persist=False)
    assert (await opted_out._knowledge_precontext(task(), state()))[0]['passages'] == []
    await opted_out._knowledge_windows(document(), task(), 'run-disabled', state=state())
    assert opted_out.knowledge.search('续航', collection='run:run-disabled')
    disabled = worker(tmp_path, enabled=False)
    assert await disabled._knowledge_windows(document(), task(), 'run', state=state()) == []
    assert await disabled._knowledge_precontext(task(), state()) == ({}, [], {})


@pytest.mark.parametrize('updates', [
    {'http_status': 500}, {'content': '  '}, {'retrieval_method': 'academic_metadata'},
    {'retrieval_method': 'academic_abstract'}, {'knowledge_ref': {'snapshot_id': 'cached'}},
    {'final_url': 'file:///generated-report.md'}, {'fetched_at': 'invalid'},
])
def test_only_original_successful_fetches_are_publishable(updates):
    assert capture_metadata(document(**updates), state()) is None


@pytest.mark.asyncio
async def test_recapture_version_chain_and_late_fetch_never_mixes_offsets(tmp_path):
    instance = worker(tmp_path)
    saved = []
    for text, hour in [('产品续航为10小时。', '08'), ('产品续航为20小时。', '09'), ('产品续航为10小时。', '10')]:
        doc = document(text, f'2026-10-24T{hour}:00:00+00:00')
        windows = await instance._knowledge_windows(doc, task(), 'run', state=state())
        assert windows and all(doc.content[a:b] == value for a,b,value,_ in windows)
        saved.append(instance.knowledge.search('续航', collection=research_collection(state()))[0]['snapshot_id'])
    assert len(set(saved)) == 3
    newest = instance.knowledge.snapshot(saved[-1], collection=research_collection(state()))
    assert newest['supersedes'] == saved[-2]
    assert instance.knowledge.snapshot(saved[0], collection=research_collection(state()))['status'] == 'archived'
    # Older network operation completes late with different content and offsets.
    windows = await instance._knowledge_windows(document('旧内容续航为5小时。', '2026-10-24T07:00:00+00:00'),
                                               task(), 'late-run', state=state('late-run'))
    assert windows == []
    assert instance.knowledge.search('续航', collection=research_collection(state()))[0]['snapshot_id'] == saved[-1]


@pytest.mark.asyncio
async def test_existing_library_and_scoped_sources_share_budget_and_policy(tmp_path):
    instance = worker(tmp_path)
    instance.runtime_config.knowledge.precontext_max_chars = 2000
    instance.runtime_config.knowledge.top_k = 2
    instance.knowledge.ingest('产品续航测试方法见说明书。', uri='file:///manual.md', title='说明书')
    await instance._knowledge_windows(document(), task(), 'run-1', state=state())
    pack, candidates, _ = await instance._knowledge_precontext(task(), state('run-2'))
    assert {p['collection'] for p in pack['passages']} == {'library', research_collection(state())}
    assert candidates and len(pack['passages']) <= 2
    assert len(json.dumps(pack, ensure_ascii=False)) <= 2000
    blocked = {**state(), 'request': {'constraints': {'blocked_domains': ['example.org']}}}
    pack, _, _ = await instance._knowledge_precontext(task(), blocked)
    assert all(p['source_uri'].startswith('file:') for p in pack['passages'])


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['search', 'snapshot'])
async def test_scoped_retrieval_failure_keeps_imported_library(tmp_path, operation):
    instance = worker(tmp_path)
    instance.knowledge.ingest('产品续航方法。', uri='file:///manual.md', title='说明')
    await instance._knowledge_windows(document(), task(), 'run-1', state=state())
    original = getattr(instance.knowledge, operation)
    def partially_broken(*args, collection='library', **kwargs):
        if collection.startswith('research:'):
            raise TimeoutError('private index offline')
        return original(*args, collection=collection, **kwargs)
    setattr(instance.knowledge, operation, partially_broken)
    pack, candidates, _ = await instance._knowledge_precontext(task(), state())
    assert pack['status'] == 'retrieved' and candidates
    assert pack['retrieval_warnings'][0]['error_type'] == 'TimeoutError'
    assert all(p['collection'] == 'library' for p in pack['passages'])


@pytest.mark.asyncio
@pytest.mark.parametrize('prefetch', [False, True])
async def test_two_service_runs_publish_then_reuse_after_restart(tmp_path, monkeypatch, prefetch):
    import deepresearch.knowledge as knowledge
    from deepresearch.service import ResearchService
    from tests.fakes import FakeFetcher, FakeSearchBackend
    config = load_runtime_config()
    config.knowledge.enabled = config.knowledge.persist_fetched_sources = True
    config.knowledge.backend = 'sqlite'
    config.graph_engineering.fetch_prefetch_enabled = prefetch
    config.mcp.enabled = config.memory.enabled = False
    def reopened(*args, **kwargs):
        return KnowledgeStore(tmp_path / 'service.sqlite', provider=CountingProvider())
    monkeypatch.setattr(knowledge, 'create_store', reopened)
    strategies_by_run = []
    for run_id in ('first-source-run', 'second-source-run'):
        llm = FakeLLM()
        async with ResearchService.in_memory(llm=llm, search_backend=FakeSearchBackend(),
            fetcher=FakeFetcher(), artifact_root=tmp_path / 'artifacts', runtime_config=config) as service:
            await service.run({'question': 'What is the verified fact?', 'run_id': run_id,
                              'options': {'max_rounds': 1, 'max_workers': 1}})
        strategies_by_run.append([json.loads(u) for kind, _, u in llm.calls
            if kind == 'json' and json.loads(u).get('phase') == 'strategy'])
    assert not any(strategy.get('knowledge_precontext', {}).get('passages') for strategy in strategies_by_run[0])
    assert any(p['source_kind'] == 'fetched_research_source' for strategy in strategies_by_run[1]
               for p in strategy.get('knowledge_precontext', {}).get('passages', []))
    assert list((tmp_path / 'artifacts').rglob('report.md'))


@pytest.mark.asyncio
@pytest.mark.parametrize('broken', ['initialization', 'ingestion'])
async def test_optional_storage_outage_does_not_break_research(tmp_path, monkeypatch, broken):
    import deepresearch.knowledge as knowledge
    from deepresearch.service import ResearchService
    from tests.fakes import FakeFetcher, FakeSearchBackend
    db = KnowledgeStore(tmp_path / 'outage.sqlite', provider=CountingProvider())
    if broken == 'initialization':
        def unavailable(*args, **kwargs):
            raise ConnectionError('database offline')
        monkeypatch.setattr(knowledge, 'create_store', unavailable)
    else:
        def unavailable(*args, **kwargs):
            raise TimeoutError('database write failed')
        db.ingest = unavailable
        monkeypatch.setattr(knowledge, 'create_store', lambda *args, **kwargs: db)
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.persist_fetched_sources = True
    config.mcp.enabled = config.memory.enabled = False
    async with ResearchService.in_memory(llm=FakeLLM(), search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(), artifact_root=tmp_path / 'artifacts', runtime_config=config) as service:
        await service.run({'question': 'What is the verified fact?', 'run_id': 'storage-outage',
                          'options': {'max_rounds': 1, 'max_workers': 1}})
    outputs = '\n'.join(p.read_text() for p in (tmp_path / 'artifacts').rglob('*.json'))
    assert 'Knowledge unavailable' in outputs if broken == 'initialization' else 'Document RAG fallback' in outputs
    assert list((tmp_path / 'artifacts').rglob('report.md'))
