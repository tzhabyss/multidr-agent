"""Real Milvus Lite storage tests. Hash vectors test mechanics, not quality."""
import json
import os
import socket
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

pytest.importorskip('filelock')

from deepresearch.knowledge.milvus_store import MilvusStore, IngestionBusy
from deepresearch.memory_system.embeddings import HashEmbeddingProvider


class CountingProvider(HashEmbeddingProvider):
    def __init__(self):
        super().__init__(32)
        self.calls = 0
    def encode_documents(self, values):
        self.calls += 1
        return super().encode_documents(values)


@pytest.fixture(scope='module')
def vector_uri(tmp_path_factory):
    pytest.importorskip('pymilvus')
    pytest.importorskip('milvus_lite')
    from milvus_lite.server import Server
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server = Server(str(tmp_path_factory.mktemp('single-milvus')/'test.db'), address=f'127.0.0.1:{port}')
    assert server.init() and server.start()
    try:
        yield f'http://127.0.0.1:{port}'
    finally:
        server.stop()


@pytest.fixture
def store(vector_uri, tmp_path):
    result = MilvusStore(milvus_uri=vector_uri, namespace='test_'+uuid.uuid4().hex,
        lock_dir=tmp_path, provider=CountingProvider(), allow_degraded=True)
    yield result
    names = {j.get('vector_collection') for j in result.records.list('job')}
    for name in names-{None}:
        if result.vectors.has_collection(collection_name=name):
            result.vectors.drop_collection(collection_name=name)
    result.vectors.drop_collection(collection_name=result.records.name)
    result.close()


def seed(store):
    return store.ingest('旧版费用100元。', uri='file:///old.md', title='制度', document_id='policy')


def test_default_factory_uses_no_pymongo(store, vector_uri, tmp_path, monkeypatch):
    from deepresearch.knowledge import create_store
    from deepresearch.runtime_config import KnowledgeRuntimeConfig
    import builtins
    original = builtins.__import__
    def guarded(name, *a, **kw):
        if name == 'pymongo' or name.startswith('pymongo.'):
            pytest.fail('single Milvus must not import MongoDB')
        return original(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guarded)
    config = KnowledgeRuntimeConfig(milvus_uri=vector_uri, milvus_namespace=store.namespace,
                                    milvus_lock_dir=str(tmp_path))
    assert config.backend == 'milvus'
    instance = create_store(config, provider=store.provider)
    assert isinstance(instance, MilvusStore) and not hasattr(instance, 'mongo')
    # Client aliases may be shared; fixture owns closing the endpoint.


def test_versions_dedup_original_scope_and_time(store):
    old = seed(store)
    cutoff = store.snapshot(old['snapshot_id'], collection='library')['published_at']
    calls = store.provider.calls
    assert seed(store)['duplicate'] and store.provider.calls == calls
    new = store.ingest('新版费用200元。', uri='file:///new.md', title='制度', document_id='policy')
    assert '200' in store.search('费用', mode='bm25')[0]['content']
    assert '100' in store.search('费用', mode='bm25', as_of=cutoff)[0]['content']
    assert seed(store)['status'] == 'archived'
    assert store.snapshot(old['snapshot_id'], collection='library')['status'] == 'archived'
    assert store.snapshot(new['snapshot_id'], collection='library')['status'] == 'active'
    assert not store.search('费用', collection='other', mode='bm25')
    with pytest.raises(ValueError):
        store.snapshot(new['snapshot_id'], collection='other')


def test_upload_hash_alias_and_forced_identity(store, tmp_path):
    a = tmp_path/'a.txt'; b = tmp_path/'b.txt'
    a.write_text('报销期限十五天。'); b.write_bytes(a.read_bytes())
    first = store.ingest_file(a); calls = store.provider.calls
    assert store.ingest_file(b)['snapshot_id'] == first['snapshot_id']
    assert store.provider.calls == calls
    second = store.ingest_file(b, document_id='independent')
    assert second['snapshot_id'] != first['snapshot_id']


def test_large_unicode_original_is_paged_and_survives_file_removal(store, tmp_path):
    content = '# 大文档\n\n' + ('段落包含中文与emoji🙂。' * 6000)
    path = tmp_path/'large.txt'; path.write_text(content)
    item = store.ingest_file(path); path.unlink()
    version = store.snapshot(item['snapshot_id'], collection='library')
    assert version['content'] == content and len(version['chunks']) > 128
    hit = store.search('emoji', mode='bm25')[0]
    assert hit['content'] == content[hit['start']:hit['end']]


@pytest.mark.parametrize('stage', ['snapshot_written', 'vector_batch_written', 'ready'])
def test_failed_new_version_keeps_old_and_retry_uses_new_id(store, stage):
    old = seed(store)
    def fail(current, job):
        if current == stage:
            raise OSError('injected storage failure')
    store._stage = fail
    with pytest.raises(OSError):
        store.ingest('新版费用200元。', uri='file:///new.md', title='制度', document_id='policy')
    job = next(j for j in store.jobs() if j['status'] == 'failed')
    assert store.snapshot(old['snapshot_id'], collection='library')['status'] == 'active'
    assert '100' in store.search('费用', mode='bm25')[0]['content']
    if job.get('vector_collection'):
        assert not store.vectors.get(collection_name=job['vector_collection'], ids=[job['snapshot_id']+':0'])
    store._stage = lambda *a: None
    new = store.retry(job['_id'])
    assert new['snapshot_id'] != job['snapshot_id'] and '200' in store.search('费用', mode='bm25')[0]['content']


@pytest.mark.parametrize('stage', ['head_published', 'old_archived'])
def test_post_commit_error_never_compensates_published_data(store, stage):
    old = seed(store)
    def fail(current, job):
        if current == stage:
            raise OSError('lost acknowledgement')
    store._stage = fail
    with pytest.raises(OSError):
        store.ingest('新版费用200元。', uri='file:///new.md', title='制度', document_id='policy')
    job = next(j for j in store.jobs() if j['status'] == 'pending')
    assert '200' in store.search('费用', mode='bm25')[0]['content']
    store._stage = lambda *a: None
    result = store.reconcile()
    assert not result['errors'] and result['published'] == 1
    with pytest.raises(ValueError, match='published'):
        store._compensate(job)
    assert store.snapshot(old['snapshot_id'], collection='library')['status'] == 'archived'


def test_actual_process_restart_and_durable_recovery(store, vector_uri, tmp_path):
    seed(store)
    script = '''
import os,sys
from deepresearch.knowledge.milvus_store import MilvusStore
from deepresearch.memory_system.embeddings import HashEmbeddingProvider
s=MilvusStore(milvus_uri=sys.argv[1],namespace=sys.argv[2],lock_dir=sys.argv[3],provider=HashEmbeddingProvider(32),allow_degraded=True)
s._stage=lambda stage,job: os._exit(23) if stage==sys.argv[4] else None
s.ingest('新版费用200元。',uri='file:///crash.md',title='制度',document_id='policy')
'''
    for stage in ['vector_batch_written', 'head_published']:
        child = subprocess.run([sys.executable, '-c', script, vector_uri, store.namespace, str(tmp_path), stage],
                               capture_output=True, timeout=45)
        assert child.returncode == 23, child.stderr.decode()
        assert not store.reconcile()['errors']
        expected = '100' if stage == 'vector_batch_written' else '200'
        assert expected in store.search('费用', mode='bm25')[0]['content']


def test_other_local_writer_rejected_readers_see_complete_old_version(store, vector_uri, tmp_path):
    seed(store)
    other = MilvusStore(milvus_uri=vector_uri, namespace=store.namespace, lock_dir=tmp_path,
                        provider=store.provider, allow_degraded=True)
    reached, release = Event(), Event()
    def pause(stage, job):
        if stage == 'ready':
            reached.set(); assert release.wait(15)
    store._stage = pause
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(store.ingest, '新版费用200元。', uri='file:///new', title='制度', document_id='policy')
        try:
            assert reached.wait(15)
            assert '100' in other.search('费用', mode='bm25')[0]['content']
            with pytest.raises(IngestionBusy):
                other.ingest('竞争写入', uri='file:///race', title='制度')
        finally:
            release.set()
        assert result.result()['snapshot_id']
    assert '200' in other.search('费用', mode='bm25')[0]['content']


def test_reindex_keeps_source_and_failed_reindex_keeps_projection(store):
    item = seed(store); before = store.snapshot(item['snapshot_id'], collection='library')
    def fail(stage, job):
        if stage == 'reindex_batch_written':
            raise OSError('rebuild failure')
    store._stage = fail
    with pytest.raises(OSError):
        store.reindex()
    assert store.snapshot(item['snapshot_id'], collection='library') == before
    store._stage = lambda *a: None
    assert store.reindex()['reindexed_snapshots'] == 1
    after = store.snapshot(item['snapshot_id'], collection='library')
    assert after['content'] == before['content'] and after['hash'] == before['hash']
    assert after['vector_collection'] != before['vector_collection']


def test_document_descriptions_and_neighbors_never_become_source(store):
    from deepresearch.knowledge.document_context import DocumentContextIndex
    from tests.test_knowledge_document_context import context_cache
    content = 'python -m tool resume run42'
    store.document_context = DocumentContextIndex(context_cache(content, '恢复研究任务的命令。'))
    store.ingest(content, uri='file:///cmd', title='命令')
    store.reranker_neighbor_chars = 8
    def tamper(query, hits):
        return [{**h, 'content': 'fabricated'} for h in reversed(hits)]
    store.reranker = tamper
    hit = store.search('恢复研究', mode='bm25')[0]
    assert hit['content'] == content and hit['rerank_status'] == 'applied'
    assert '_rerank_neighbors' not in hit
    store.document_context = None
    store.reindex()
    assert not store.search('恢复研究', mode='bm25')


@pytest.mark.parametrize('suffix,data', [('.pdf', b'not PDF'), ('.txt', b'  '), ('.exe', b'bad')])
def test_failed_parse_is_journalled(store, tmp_path, suffix, data):
    seed(store); path = tmp_path/('bad'+suffix); path.write_bytes(data)
    with pytest.raises(Exception):
        store.ingest_file(path, document_id='policy')
    assert any(j['phase'] == 'parse_failed' for j in store.jobs())
    assert '100' in store.search('费用', mode='bm25')[0]['content']


def test_fallback_and_source_hash_verification(store):
    item = seed(store)
    store.reranker = lambda *a: (_ for _ in ()).throw(TimeoutError())
    trace = {}
    hit = store.search('费用', candidate_policy='quota_dense', trace=trace)[0]
    assert hit['vector_status'] == 'fallback_bm25' and hit['rerank_status'] == 'fallback'
    assert trace['policy'] == 'quota_dense'
    value = store.records.get('version', item['snapshot_id'])
    value['content'] = 'tampered'
    store.records.put('version', item['snapshot_id'], value)
    with pytest.raises(ValueError, match='hash'):
        store.search('费用', mode='bm25')


def test_unavailable_semantic_model_does_not_publish_hash_index(store):
    seed(store); store.allow_degraded = False
    with pytest.raises(ValueError, match='semantic'):
        store.ingest('新版费用200元。', uri='file:///new', title='制度', document_id='policy')
    assert '100' in store.search('费用', mode='bm25')[0]['content']


@pytest.mark.asyncio
async def test_worker_precontext_reuses_sources_after_reopen(store, vector_uri, tmp_path):
    from tests.test_knowledge_research_sources import worker, task, state, document
    instance = worker(tmp_path)
    instance.knowledge = store
    assert await instance._knowledge_windows(document(), task(), 'run-1', state=state())
    calls = store.provider.calls
    await instance._knowledge_windows(document(captured='2026-10-24T09:00:00+00:00'), task(), 'run-2', state=state('run-2'))
    assert store.provider.calls == calls
    reopened = MilvusStore(milvus_uri=vector_uri, namespace=store.namespace, lock_dir=tmp_path,
                           provider=store.provider, allow_degraded=True)
    instance.knowledge = reopened
    pack, _, _ = await instance._knowledge_precontext(task(), state('run-3'))
    assert pack['status'] == 'retrieved' and pack['passages'][0]['content'] in document().content
    assert not reopened.search('续航', collection='library', mode='bm25')


def test_http_capture_order_and_returning_old_content_create_new_snapshot(store):
    a = store.ingest('续航10小时', uri='https://example.org/a', title='产品', refresh_capture=True,
                     metadata={'last_fetched_at': '2026-10-24T01:00:00+00:00'})
    b = store.ingest('续航12小时', uri='https://example.org/a', title='产品', refresh_capture=True,
                     metadata={'last_fetched_at': '2026-10-24T02:00:00+00:00'})
    assert store.ingest('续航10小时', uri='https://example.org/a', title='产品', refresh_capture=True,
                       metadata={'last_fetched_at': '2026-10-24T01:00:00+00:00'})['superseded_capture']
    c = store.ingest('续航10小时', uri='https://example.org/a', title='产品', refresh_capture=True,
                     metadata={'last_fetched_at': '2026-10-24T03:00:00+00:00'})
    assert len({a['snapshot_id'], b['snapshot_id'], c['snapshot_id']}) == 3


@pytest.mark.skipif(not os.environ.get('RAG_MIGRATION_MONGO_URI'), reason='explicit disposable legacy Mongo URI required')
def test_real_legacy_migration_preserves_ids_history_and_resumes(store):
    from deepresearch.knowledge.dual_store import MongoMilvusStore
    from deepresearch.knowledge.migration import migrate_collection
    legacy = MongoMilvusStore(os.environ['RAG_MIGRATION_MONGO_URI'], database='migration_test_'+uuid.uuid4().hex,
        vector_client=store.vectors, provider=store.provider, allow_degraded=True)
    try:
        old = seed(legacy)
        new = legacy.ingest('新版费用200元。', uri='file:///new', title='制度', document_id='policy')
        cutoff = legacy.snapshot(old['snapshot_id'], collection='library')['published_at'].replace(tzinfo=__import__('datetime').timezone.utc).isoformat()
        heads_before = list(legacy.db.heads.find({}))
        result = migrate_collection(legacy, store)
        assert result['migrated_snapshots'] == 2
        assert store.snapshot(old['snapshot_id'], collection='library')['status'] == 'archived'
        assert store.search('费用', mode='bm25')[0]['snapshot_id'] == new['snapshot_id']
        assert store.search('费用', mode='bm25', as_of=cutoff)[0]['snapshot_id'] == old['snapshot_id']
        assert migrate_collection(legacy, store)['verified_existing'] == 2
        assert list(legacy.db.heads.find({})) == heads_before
        # Preserved IDs keep saved citation references valid.
        expected = list(legacy.db.chunks.find({'snapshot': new['snapshot_id']}).sort('start', 1))
        assert store.records.get('version', new['snapshot_id'])['chunks'] == expected
        broken = store.records.get('version', new['snapshot_id'])
        broken['content'] = 'tampered'
        store.records.put('version', new['snapshot_id'], broken)
        with pytest.raises(ValueError, match='hash'):
            migrate_collection(legacy, store)
    finally:
        for name in {v.get('vector_collection') for v in legacy.db.versions.find({})}-{None}:
            if store.vectors.has_collection(collection_name=name):
                store.vectors.drop_collection(collection_name=name)
        legacy.mongo.drop_database(legacy.db.name)
        legacy.mongo.close()


def test_missing_original_page_fails_closed(store):
    item = store.ingest('原文🙂。'*5000, uri='file:///pages', title='长文')
    rid = store.records.key('version', item['snapshot_id'])
    raw = store.records._rows([rid])[rid]
    page = json.loads(raw)['pages'][0]
    store.vectors.delete(collection_name=store.records.name, ids=[page])
    store.vectors.flush(collection_name=store.records.name)
    with pytest.raises(ValueError, match='incomplete'):
        store.search('原文', mode='bm25')
    # Teardown lists jobs, not the intentionally corrupted version.


def test_unconfirmed_head_write_keeps_staging_until_reconcile(store, monkeypatch):
    seed(store)
    original = store.records.put
    def offline(kind, key, value):
        if kind == 'head':
            raise TimeoutError('publication transport failure')
        return original(kind, key, value)
    monkeypatch.setattr(store.records, 'put', offline)
    with pytest.raises(TimeoutError):
        store.ingest('新版费用200元。', uri='file:///new', title='制度', document_id='policy')
    job = next(j for j in store.jobs() if j['status'] == 'pending')
    assert job['phase'] == 'publication_unknown'
    assert store.vectors.get(collection_name=job['vector_collection'], ids=[job['snapshot_id']+':0'])
    assert '100' in store.search('费用', mode='bm25')[0]['content']
    monkeypatch.setattr(store.records, 'put', original)
    with pytest.raises(IngestionBusy, match='reconcile'):
        store.ingest('后续写入', uri='file:///later', title='后续')
    assert store.reconcile()['compensated'] == 1


def test_rebuild_lost_publish_ack_keeps_committed_index(store, monkeypatch):
    item = seed(store)
    original = store.records.put
    once = [False]
    def lost_ack(kind, key, value):
        original(kind, key, value)
        if kind == 'version' and value.get('vector_collection') and not once[0]:
            once[0] = True
            raise TimeoutError('index confirmation lost')
    monkeypatch.setattr(store.records, 'put', lost_ack)
    with pytest.raises(TimeoutError):
        store.reindex()
    monkeypatch.setattr(store.records, 'put', original)
    version = store.snapshot(item['snapshot_id'], collection='library')
    assert store.vectors.has_collection(collection_name=version['vector_collection'])
    assert store.reconcile()['published'] == 1
    assert store.search('费用', mode='bm25')


def test_pdf_reparse_upgrade_atomicity_and_ocr_provenance(store, tmp_path, monkeypatch):
    from pathlib import Path
    from deepresearch.knowledge.context import build_context
    from deepresearch.knowledge.document_parser import IncompleteDocumentError
    path = tmp_path/'report.pdf'
    path.write_bytes((Path(__file__).parent/'fixtures/knowledge/mixed.pdf').read_bytes())
    # Simulate an already-published snapshot made by the previous text-only parser.
    original = store._parse
    monkeypatch.setattr(store, '_parse', lambda *a: ('Legacy native page budget 67890.', {'page_starts': [0]}))
    old = store.ingest_file(path)
    assert store.ingest_file(path)['duplicate']
    def failed(*a):
        raise IncompleteDocumentError('OCR unavailable')
    monkeypatch.setattr(store, '_parse', failed)
    with pytest.raises(IncompleteDocumentError):
        store.ingest_file(path, reparse=True)
    assert store.snapshot(old['snapshot_id'], collection='library')['status'] == 'active'
    failed_job = next(j for j in store.jobs() if j['phase'] == 'parse_failed')
    monkeypatch.setattr(store, '_parse', original)
    # Retry must preserve reparse=True, not short-circuit on the older byte hash.
    new = store.retry(failed_job['_id'])
    assert new['snapshot_id'] != old['snapshot_id']
    assert store.snapshot(old['snapshot_id'], collection='library')['status'] == 'archived'
    snapshot = store.snapshot(new['snapshot_id'], collection='library')
    meta = json.loads(snapshot['metadata'])
    assert meta['parse_complete'] and meta['pages'][1]['method'] == 'ocr'
    hit = store.search('12345', mode='bm25')[0]
    assert hit['page'] == 2
    assert snapshot['content'][hit['start']:hit['end']] == hit['content']
    pack, _ = build_context(store, ['12345'], search_mode='bm25')
    assert pack['passages'][0]['parsing']['ocr_used']
    assert pack['passages'][0]['parsing']['requires_original_verification']
    assert store.ingest_file(path)['snapshot_id'] == new['snapshot_id']


def test_failed_reparse_after_vector_write_can_retry_without_old_dedup(store, tmp_path):
    path = tmp_path/'doc.txt'; path.write_text('本地材料12345。')
    old = store.ingest_file(path)
    def fail(stage, job):
        if stage == 'snapshot_written':
            raise OSError('injected reparse index failure')
    store._stage = fail
    with pytest.raises(OSError):
        store.ingest_file(path, reparse=True)
    job = next(j for j in store.jobs() if j['status'] == 'failed')
    store._stage = lambda *a: None
    new = store.retry(job['_id'])
    assert new['snapshot_id'] != old['snapshot_id']
    assert store.snapshot(new['snapshot_id'], collection='library')['status'] == 'active'


@pytest.mark.parametrize('persistent', [False, True])
def test_vector_confirmation_retries_reads_but_rejects_persistent_missing_rows(store, monkeypatch, persistent):
    original = store.vectors.get
    misses = []
    def lagged(*args, **kwargs):
        if kwargs.get('collection_name', '').startswith('rag_vectors_'):
            if persistent or not misses:
                misses.append(True)
                return []
        return original(*args, **kwargs)
    monkeypatch.setattr(store.vectors, 'get', lagged)
    if persistent:
        with pytest.raises(ValueError, match='index is incomplete'):
            seed(store)
        assert len(misses) == 6
        assert not store.records.list('head')
    else:
        item = seed(store)
        assert len(misses) == 1
        assert store.snapshot(item['snapshot_id'], collection='library')['status'] == 'active'
