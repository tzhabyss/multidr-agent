"""Retrieval-specific model presets; memory embeddings remain independent."""
from ..memory_system.embeddings import SentenceTransformerEmbeddingProvider
from .store import digest

MODEL_PRESETS = {
    'intfloat/multilingual-e5-small': {'dimensions': 384, 'query_prefix': 'query: ', 'document_prefix': 'passage: '},
    'BAAI/bge-small-zh-v1.5': {'dimensions': 512, 'query_prefix': '为这个句子生成表示以用于检索相关文章：', 'document_prefix': ''},
    'thenlper/gte-small-zh': {'dimensions': 512, 'query_prefix': '', 'document_prefix': ''},
}


class RetrievalEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    def __init__(self, model_name, *, revision=None, device=None):
        if model_name not in MODEL_PRESETS:
            raise ValueError(f'unsupported retrieval model; choose one of {list(MODEL_PRESETS)}')
        super().__init__(model_name, **MODEL_PRESETS[model_name])
        self.revision, self.device = revision, device

    @property
    def model_id(self):
        identity = f'{self.query_prefix}|{self.document_prefix}|{self.revision or "unversioned"}'
        return f'sentence-transformers:{self.model_name}:{digest(identity)[:12]}'

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name, revision=self.revision,
                        device=self.device, trust_remote_code=False)
                    getter = getattr(self._model, 'get_embedding_dimension', None) or self._model.get_sentence_embedding_dimension
                    self._dimensions = getter()
        return self._model
