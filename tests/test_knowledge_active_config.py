"""Validate the explicitly enabled deployment and reproducible reranker loading."""
from pathlib import Path
from deepresearch.runtime_config import load_runtime_config
from deepresearch.knowledge.reranker import CrossEncoderReranker


def test_enabled_deployment_selects_pinned_cpu_models_and_bounded_budgets():
    c = load_runtime_config(Path(__file__).resolve().parents[1]/'config/runtime.yaml').knowledge
    assert c.enabled and c.backend == 'milvus'
    assert c.embedding_model == 'thenlper/gte-small-zh' and c.embedding_device == 'cpu'
    assert len(c.embedding_revision) == 40
    assert c.reranker_model == 'BAAI/bge-reranker-v2-m3' and c.reranker_device == 'cpu'
    assert len(c.reranker_revision) == 40 and c.reranker_max_length == 512
    assert c.reranker_timeout_seconds < c.timeout_seconds <= 120
    assert c.candidate_policy == 'quota_dense' and c.document_context_path
    assert not c.query_rewrite_enabled and c.reranker_neighbor_chars == 0


def test_reranker_loads_requested_revision_device_and_length_and_reports_completed_calls(monkeypatch):
    c = load_runtime_config(Path(__file__).resolve().parents[1]/'config/runtime.yaml').knowledge
    loaded = []
    class Model:
        def predict(self, pairs):
            assert pairs == [('query', 'title\n\nfirst'), ('query', 'title\n\nsecond')]
            return [0.1, 0.9]
    def load(name, **kwargs):
        loaded.append((name, kwargs)); return Model()
    monkeypatch.setattr('deepresearch.knowledge.reranker.load_cross_encoder', load)
    ranker = CrossEncoderReranker.from_config(c)
    try:
        hits = [{'title': 'title', 'content': 'first'}, {'title': 'title', 'content': 'second'}]
        assert ranker('query', hits) == list(reversed(hits))
        assert ranker('query', hits) == list(reversed(hits))
        assert len(loaded) == 1
        assert loaded[0] == (c.reranker_model, {'revision': c.reranker_revision, 'device': 'cpu',
                                              'trust_remote_code': False, 'max_length': 512})
        assert ranker.completed_calls == 2
    finally:
        ranker.executor.shutdown(wait=True)
