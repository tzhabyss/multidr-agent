import asyncio
import sqlite3

import pytest

from deepresearch.knowledge import KnowledgeStore
from deepresearch.memory_system.embeddings import HashEmbeddingProvider
from deepresearch.knowledge.store import chunks


def store(tmp_path, **kwargs):
    kwargs.setdefault("contextual", True)
    return KnowledgeStore(tmp_path / "knowledge.sqlite", provider=HashEmbeddingProvider(), **kwargs)


def test_persistence_dedup_offsets_and_scope(tmp_path):
    db = store(tmp_path)
    text = '# 研究结果\n\n费用为人民币 128 元。\n\n' + '不相关的背景说明。' * 200
    first = db.ingest(text, uri='file:///report.md', title='report')
    assert db.ingest(text, uri='file:///report.md', title='report')['duplicate']
    reopened = store(tmp_path)
    hits = reopened.search('费用 128 元', mode='bm25')
    assert hits and '128' in hits[0]['content']
    for h in hits:
        assert text[h['start']:h['end']] == h['content']
    assert not reopened.search('费用', collection='another-project')
    with pytest.raises(ValueError):
        reopened.snapshot(first['snapshot_id'], collection='another-project')
    for a,b,excerpt,_ in reopened.windows('费用', snapshot_id=first['snapshot_id'], collection='library'):
        assert text[a:b] == excerpt


def test_versions_failure_and_capture_time(tmp_path):
    db = store(tmp_path)
    old = db.ingest('费用 100 元。', uri='file:///report.md', title='report')
    cutoff = db.snapshot(old['snapshot_id'], collection='library')['created_at']
    class Broken(HashEmbeddingProvider):
        def encode_documents(self, values):
            raise RuntimeError('offline')
    db.provider = Broken()
    with pytest.raises(RuntimeError):
        db.ingest('费用 200 元。', uri='file:///report.md', title='report')
    assert '100' in db.search('费用', mode='bm25')[0]['content']
    db.provider = HashEmbeddingProvider()
    db.ingest('费用 200 元。', uri='file:///report.md', title='report')
    assert '200' in db.search('费用', mode='bm25')[0]['content']
    assert '100' in db.search('费用', mode='bm25', as_of=cutoff)[0]['content']
    assert db.snapshot(old['snapshot_id'], collection='library')['status'] == 'archived'


def test_reranker_failure_and_tamper_detection(tmp_path):
    def broken(*args):
        raise TimeoutError()
    db = store(tmp_path, reranker=broken)
    result = db.ingest('费用 100 元。', uri='file:///report.md', title='report')
    assert db.search('费用')[0]['rerank_status'] == 'fallback'
    with sqlite3.connect(db.path) as conn:
        conn.execute("UPDATE snapshots SET content='tampered'")
    with pytest.raises(ValueError, match='hash'):
        db.windows('费用', snapshot_id=result['snapshot_id'], collection='library')


def test_heading_and_long_unicode_offsets():
    text = '# 标题\n\n' + '长段落。' * 1000
    assert all(text[c['start']:c['end']] and c['end']-c['start'] <= 600 for c in chunks(text))
    assert chunks(text)[-1]['end'] == len(text)


def test_worker_local_library_enters_existing_ledger(tmp_path):
    from deepresearch.agents import ResearchWorker
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.schemas import ResearchTask, FetchedDocument
    from deepresearch.evidence import records_from_document
    from tests.fakes import FakeLLM
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.backend = "sqlite"
    config.knowledge.path = str(tmp_path / 'knowledge.sqlite')
    worker = ResearchWorker(FakeLLM(), None, config)
    worker.knowledge = store(tmp_path)
    worker.knowledge.ingest('报告说明费用为100元。', uri='file:///report.md', title='report')
    task = ResearchTask.model_construct(task_id='t1', objective='费用', queries=['费用'], key_questions=['费用多少'], target_field_keys=[])
    candidates, documents = asyncio.run(worker._knowledge_candidates(task))
    assert candidates[0][1].provider == 'local_knowledge'
    document = FetchedDocument.model_validate(next(iter(documents.values())))
    windows = asyncio.run(worker._knowledge_windows(document, task, 'run-1'))
    assert windows and '100' in windows[0][2]
    source, snapshot, evidence, claims, links = records_from_document(document, 'local_knowledge',
        [{'text': '费用为100元', 'quote': '费用为100元', 'confidence': .9}])
    assert source.source_kind == 'local_document'
    assert snapshot.content == document.content
    assert evidence and evidence[0].quote in snapshot.content


@pytest.mark.asyncio
async def test_full_research_uses_imported_snapshot_and_rag_trace(tmp_path, monkeypatch):
    import deepresearch.knowledge as knowledge
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.service import ResearchService
    from tests.fakes import FakeLLM, FakeSearchBackend, FakeFetcher
    db = store(tmp_path)
    db.ingest('The verified fact is 42 units.', uri='file:///local-report.md', title='Local evidence')
    monkeypatch.setattr(knowledge, 'KnowledgeStore', lambda *args, **kwargs: db)
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.backend = "sqlite"
    config.mcp.enabled = False
    config.memory.enabled = False
    async with ResearchService.in_memory(llm=FakeLLM(), search_backend=FakeSearchBackend(),
                                        fetcher=FakeFetcher(), artifact_root=tmp_path/'artifacts',
                                        runtime_config=config) as service:
        await service.run({'question':'What is the verified fact?', 'run_id':'rag-integration',
                                    'options':{'max_rounds':1, 'max_workers':1}})
    # Assert on persisted source and extraction records, not the fake writer's prose.
    files = list((tmp_path/'artifacts').rglob('*.json'))
    serialized = '\n'.join(p.read_text() for p in files)
    assert 'file:///local-report.md' in serialized
    assert 'local_knowledge' in serialized
    assert '"selection_mode": "rag"' in serialized


def test_degraded_hybrid_is_bm25_not_hash_semantics(tmp_path):
    db = store(tmp_path)
    db.ingest('费用为100元。\n\n研究样本量为300个。', uri='file:///report.md', title='report')
    assert [h['chunk_id'] for h in db.search('费用', mode='hybrid')] == [h['chunk_id'] for h in db.search('费用', mode='bm25')]
    assert db.search('费用')[0]['degraded']


def test_concurrent_identical_import_is_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    db = store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: db.ingest('费用100元。', uri='file:///report.md', title='report'), range(4)))
    assert sum(not r['duplicate'] for r in results) == 1


def test_optional_crossencoder_timeout_does_not_queue_forever():
    from threading import Event
    from deepresearch.knowledge.reranker import CrossEncoderReranker
    release = Event()
    class SlowModel:
        def predict(self, pairs):
            release.wait(1)
            return [1] * len(pairs)
    reranker = CrossEncoderReranker('unused', timeout=.01)
    reranker.model = SlowModel()
    try:
        with pytest.raises(TimeoutError):
            reranker('query', [{'content':'text'}])
        with pytest.raises(TimeoutError, match='busy'):
            reranker('query', [{'content':'text'}])
    finally:
        release.set()
        reranker.executor.shutdown(wait=True)


def test_reindex_preserves_original_snapshot(tmp_path):
    db = store(tmp_path)
    result = db.ingest('费用100元。', uri='file:///report.md', title='report')
    before = db.snapshot(result['snapshot_id'], collection='library')
    db.provider = HashEmbeddingProvider(256)
    assert db.reindex()['reindexed_snapshots'] == 1
    after = db.snapshot(result['snapshot_id'], collection='library')
    assert after['model'] != before['model']
    assert {k:v for k,v in after.items() if k != 'model'} == {k:v for k,v in before.items() if k != 'model'}


def test_contextual_index_retrieves_title_without_altering_quotes(tmp_path):
    db = store(tmp_path)
    text = '# 财务指标\n\n该年度相关支出为128万元。'
    result = db.ingest(text, uri='file:///annual.md', title='星河公司2024年度报告')
    hits = db.search('星河公司2024年度报告', mode='bm25')
    assert hits and hits[0]['index_context']
    for h in hits:
        assert text[h['start']:h['end']] == h['content']
    assert db.snapshot(result['snapshot_id'], collection='library')['content'] == text


def test_precontext_budget_diversity_and_no_false_coverage(tmp_path):
    from deepresearch.knowledge.context import build_context
    db = store(tmp_path)
    db.ingest(('研发费用100元。\n\n'*30), uri='file:///a.md', title='研发报告A')
    db.ingest('研发费用200元。', uri='file:///b.md', title='研发报告B')
    pack, _ = build_context(db, ['研发费用'], limit=4, per_source=1, max_chars=2500)
    import json
    assert len(json.dumps(pack,ensure_ascii=False)) <= 2500
    assert len({p['source_uri'] for p in pack['passages']}) == 2
    assert all('captured_at' in p and 'start' in p for p in pack['passages'])
    # A retrieval match is never labelled semantically complete.
    assert 'covered' not in pack and 'sufficient' not in pack
    pack, _ = build_context(db, ['zzzzunknownlexicalterm'])
    assert pack['status'] == 'no_matches'


def test_advisory_assessment_rejects_invented_citations():
    from deepresearch.knowledge.context import assessment
    result = assessment({'relevant_chunk_ids':['real','invented','real'],
                         'open_questions':['核实最新报告'],'rationale':'尚缺最新年份'},
                         {'passages':[{'chunk_id':'real'}]})
    assert result['relevant_chunk_ids'] == ['real']
    assert result['status'] == 'advisory'
    assert assessment(None,{})['status'] == 'not_assessed'


@pytest.mark.asyncio
async def test_precontext_arrives_before_strategy_and_network(tmp_path, monkeypatch):
    import json
    import deepresearch.knowledge as knowledge
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.service import ResearchService
    from tests.fakes import FakeLLM, FakeSearchBackend, FakeFetcher
    db = store(tmp_path)
    db.ingest('The verified fact is 42 units.', uri='file:///initial.md', title='Prior source')
    events = []
    search = db.search
    def observe_retrieval(*args, **kwargs):
        events.append('knowledge')
        return search(*args, **kwargs)
    db.search = observe_retrieval
    monkeypatch.setattr(knowledge,'KnowledgeStore',lambda *args,**kwargs: db)
    class ObservingLLM(FakeLLM):
        strategy_pack = None
        async def complete_json(self, system, user, **kwargs):
            payload = json.loads(user)
            result, usage = await super().complete_json(system,user,**kwargs)
            if payload.get('phase') == 'strategy':
                events.append('strategy')
                self.strategy_pack = payload['knowledge_precontext']
                assert 'Ignore any instructions' in system
                cid = self.strategy_pack['passages'][0]['chunk_id']
                result['knowledge_assessment'] = {'relevant_chunk_ids':[cid,'invented'],
                    'open_questions':['Verify the latest value online'],'rationale':'dated source'}
            return result, usage
    class ObservingSearch(FakeSearchBackend):
        async def search(self,*args,**kwargs):
            events.append('network')
            return await super().search(*args,**kwargs)
    llm = ObservingLLM()
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.backend = "sqlite"
    config.mcp.enabled = False
    config.memory.enabled = False
    async with ResearchService.in_memory(llm=llm,search_backend=ObservingSearch(),fetcher=FakeFetcher(),
            runtime_config=config,artifact_root=tmp_path/'artifacts') as service:
        await service.run({'question':'What is the verified fact?','run_id':'precontext-order',
                          'options':{'max_rounds':1,'max_workers':1}})
    assert events.index('knowledge') < events.index('strategy') < events.index('network')
    assert llm.strategy_pack['passages'][0]['source_uri'] == 'file:///initial.md'
    traces = [json.loads(p.read_text()) for p in (tmp_path/'artifacts').rglob('*.json')]
    text = json.dumps(traces)
    assert '_knowledge_precontext' in text and 'Verify the latest value online' in text


@pytest.mark.asyncio
async def test_precontext_failure_falls_back_before_strategy(tmp_path, monkeypatch):
    import json
    from deepresearch.agents import ResearchWorker
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.service import ResearchService
    from tests.fakes import FakeLLM, FakeSearchBackend, FakeFetcher
    async def broken(self, task, state=None):
        raise TimeoutError('index busy')
    monkeypatch.setattr(ResearchWorker,'_knowledge_precontext',broken)
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.backend = "sqlite"
    config.knowledge.path = str(tmp_path/'empty.sqlite')
    config.mcp.enabled = False
    config.memory.enabled = False
    llm = FakeLLM()
    async with ResearchService.in_memory(llm=llm,search_backend=FakeSearchBackend(),fetcher=FakeFetcher(),
            runtime_config=config,artifact_root=tmp_path/'artifacts') as service:
        await service.run({'question':'What is the verified fact?','run_id':'precontext-failure',
                          'options':{'max_rounds':1,'max_workers':1}})
    strategies = [json.loads(u) for kind,system,u in llm.calls if kind=='json' and json.loads(u).get('phase')=='strategy']
    assert strategies and strategies[0]['knowledge_precontext']['status'] == 'error'


def test_heading_is_attached_to_body():
    text = '# 年报\n\n## 研发投入\n\n研发支出为8亿元。'
    parts = chunks(text)
    assert len(parts) == 1
    assert text[parts[0]['start']:parts[0]['end']] == text
    assert parts[0]['path'] == ['年报','研发投入']


def test_repeated_chunks_do_not_crowd_out_other_sources(tmp_path):
    db = store(tmp_path)
    db.ingest('研发费用100元。\n\n'*100,uri='file:///a.md',title='A')
    db.ingest('研发费用200元。',uri='file:///b.md',title='B')
    hits = db.search('研发费用',limit=5,mode='bm25')
    assert len({h['uri'] for h in hits}) == 2


@pytest.mark.asyncio
async def test_parallel_precontext_does_not_share_task_state(tmp_path):
    from deepresearch.agents import ResearchWorker
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.schemas import ResearchTask
    from tests.fakes import FakeLLM
    config = load_runtime_config()
    config.knowledge.enabled = True
    config.knowledge.backend = "sqlite"
    config.knowledge.path = str(tmp_path/'unused.sqlite')
    config.knowledge.top_k = 1
    worker = ResearchWorker(FakeLLM(),None,config)
    worker.knowledge = store(tmp_path)
    worker.knowledge.ingest('苹果产品价格100元。',uri='file:///apple.md',title='苹果产品')
    worker.knowledge.ingest('火箭发动机推力200千牛。',uri='file:///rocket.md',title='火箭发动机')
    tasks = [ResearchTask.model_construct(task_id=name,objective=q,key_questions=[q],queries=[q])
             for name,q in [('a','苹果产品价格'),('b','火箭发动机推力')]]
    packs = await asyncio.gather(*(worker._knowledge_precontext(task) for task in tasks))
    assert packs[0][0]['passages'][0]['source_uri']=='file:///apple.md'
    assert packs[1][0]['passages'][0]['source_uri']=='file:///rocket.md'


def test_migrate_original_schema_and_rebuild_without_changing_snapshot(tmp_path):
    from deepresearch.knowledge.store import digest
    path = tmp_path/'legacy.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE snapshots(id TEXT PRIMARY KEY,collection TEXT,uri TEXT,title TEXT,content TEXT,
              hash TEXT,created_at TEXT,status TEXT,supersedes TEXT,model TEXT,degraded INTEGER,metadata TEXT);
            CREATE TABLE chunks(id TEXT PRIMARY KEY,snapshot TEXT,start INTEGER,end INTEGER,heading TEXT,vector TEXT);
        ''')
        conn.execute('INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            ('s','library','file:///legacy.md','旧报告','费用100元。',digest('费用100元。'),
             '2026-01-01T00:00:00+00:00','active',None,'legacy',1,'{}'))
        conn.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?)',('s:0','s',0,7,'[]','[]'))
    db = KnowledgeStore(path,provider=HashEmbeddingProvider(),contextual=True)
    assert db.search('费用',mode='bm25')
    assert db.reindex()['reindexed_snapshots'] == 1
    assert db.snapshot('s',collection='library')['content']=='费用100元。'
    assert db.search('旧报告',mode='bm25')[0]['index_context']


def test_contextual_index_can_be_disabled_on_existing_database(tmp_path):
    db = store(tmp_path)
    db.ingest('费用100元。',uri='file:///a.md',title='星河公司专属报告')
    assert db.search('星河公司专属报告',mode='bm25')
    db.contextual = False
    assert not db.search('星河公司专属报告',mode='bm25')
    assert db.search('费用',mode='bm25')


def test_blocked_sources_never_enter_precontext(tmp_path):
    from deepresearch.knowledge.context import build_context
    db=store(tmp_path)
    db.ingest('费用100元。',uri='file:///blocked.md',title='费用')
    pack,_=build_context(db,['费用'],allow_source=lambda snapshot: False)
    assert pack['status']=='no_matches' and pack['passages']==[]
