"""Persistent, source-grounded research document retrieval."""
from .store import KnowledgeStore


def create_store(config, *, path=None, provider=None, reranker=None):
    if config.backend == "mongo_milvus":
        import os
        from .dual_store import MongoMilvusStore
        from .embeddings import RetrievalEmbeddingProvider
        from .document_context import DocumentContextIndex
        if path is not None:
            raise ValueError("--db selects SQLite files; use knowledge.mongo_uri/milvus_uri for mongo_milvus")
        return MongoMilvusStore(config.mongo_uri, database=config.mongo_database,
            milvus_uri=config.milvus_uri, milvus_token=os.environ.get(config.milvus_token_env, ""),
            provider=provider or RetrievalEmbeddingProvider(config.embedding_model, revision=config.embedding_revision),
            reranker=reranker, contextual=config.contextual_index, parser=config.parser,
            candidate_policy=config.candidate_policy,
            reranker_neighbor_chars=config.reranker_neighbor_chars,
            document_context=DocumentContextIndex.load(config.document_context_path) if config.document_context_path else None)
    if config.document_context_path:
        raise ValueError("document descriptions require the mongo_milvus backend")
    if config.reranker_neighbor_chars:
        raise ValueError("reranker source neighbors require the mongo_milvus backend")
    if config.candidate_policy != "rrf":
        raise ValueError("candidate quotas require the mongo_milvus backend")
    return KnowledgeStore(path or config.path, provider=provider, reranker=reranker,
                          contextual=config.contextual_index)


__all__ = ["KnowledgeStore", "create_store"]
