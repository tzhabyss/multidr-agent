import pytest

from deepresearch.knowledge.dual_store import MongoMilvusStore
from deepresearch.knowledge.reranker import document_for_rerank


def target():
    return {'chunk_id': 'c', 'snapshot_id': 's', 'title': '文档', 'heading': ['章节'],
            'content': '目标原文', 'start': 5, 'end': 9}


def test_source_windows_are_bounded_scoped_and_copied():
    store = MongoMilvusStore.__new__(MongoMilvusStore)
    store.reranker_neighbor_chars = 3
    reads = []
    def snapshot(sid, *, collection):
        reads.append((sid, collection))
        return {'content': 'abcde目标原文fghij'}
    store.snapshot = snapshot
    hit = target()
    ranked = store.prepare_rerank_hits([hit, dict(hit)], collection='private')
    assert reads == [('s', 'private')]
    assert ranked[0]['_rerank_neighbors'] == {'before': 'cde', 'after': 'fgh'}
    assert ranked[0]['content'] == hit['content'] and '_rerank_neighbors' not in hit
    ranked[0]['_rerank_neighbors']['before'] = 'tampered'
    assert ranked[1]['_rerank_neighbors']['before'] == 'cde'


def test_disabled_neighbors_do_not_read_source_or_accept_injected_neighbors():
    store = MongoMilvusStore.__new__(MongoMilvusStore)
    store.reranker_neighbor_chars = 0
    store.snapshot = lambda *a, **k: pytest.fail('must not load disabled source neighbors')
    hit = {**target(), '_rerank_neighbors': {'before': 'injected', 'after': ''}}
    assert '_rerank_neighbors' not in store.prepare_rerank_hits([hit])[0]


def test_window_preparation_rejects_content_not_matching_snapshot():
    store = MongoMilvusStore.__new__(MongoMilvusStore)
    store.reranker_neighbor_chars = 120
    store.snapshot = lambda *a, **k: {'content': 'different source'}
    with pytest.raises(ValueError, match='immutable source'):
        store.prepare_rerank_hits([target()])


def test_optional_neighbors_follow_the_original_target():
    hit = target()
    plain = document_for_rerank(hit)
    assert plain == '文档\n章节\n目标原文'
    hit['_rerank_neighbors'] = {'before': '前置说明', 'after': '后置说明'}
    expanded = document_for_rerank(hit)
    assert expanded.startswith(plain)
    assert expanded.index('目标原文') < expanded.index('前置说明') < expanded.index('后置说明')
