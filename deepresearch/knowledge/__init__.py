"""Persistent, source-grounded research document retrieval."""
import logging

from .store import KnowledgeStore

logger = logging.getLogger(__name__)


def create_store(config, *, path=None, provider=None, reranker=None):
    from .document_parser import ParseOptions
    parse_options = ParseOptions(ocr_backend=config.ocr_backend, ocr_languages=config.ocr_languages,
        ocr_timeout_seconds=config.ocr_timeout_seconds, parse_timeout_seconds=config.parse_timeout_seconds,
        max_pages=config.parse_max_pages, allow_partial=config.parse_allow_partial)
    if config.backend in {"milvus", "mongo_milvus"}:
        import os
        from .embeddings import RetrievalEmbeddingProvider
        from .document_context import DocumentContextIndex
        if path is not None:
            raise ValueError("--db selects SQLite files; use knowledge.milvus_uri for Milvus backends")
        document_context = None
        if config.document_context_path:
            try:
                document_context = DocumentContextIndex.load(config.document_context_path)
            except FileNotFoundError:
                # Generated descriptions are optional local artifacts, not required
                # checkout contents. Invalid caches must still fail validation.
                logger.warning("Retrieval description cache is missing; using original-text indexing")
        kwargs = dict(milvus_uri=config.milvus_uri, milvus_token=os.environ.get(config.milvus_token_env, ""),
            provider=provider or RetrievalEmbeddingProvider(config.embedding_model, revision=config.embedding_revision,
                                                           device=config.embedding_device),
            reranker=reranker, contextual=config.contextual_index, parser=config.parser,
            parse_options=parse_options,
            candidate_policy=config.candidate_policy,
            reranker_neighbor_chars=config.reranker_neighbor_chars,
            document_context=document_context)
        if config.backend == "milvus":
            from .milvus_store import MilvusStore
            return MilvusStore(namespace=config.milvus_namespace, lock_dir=config.milvus_lock_dir, **kwargs)
        from .dual_store import MongoMilvusStore
        return MongoMilvusStore(config.mongo_uri, database=config.mongo_database, **kwargs)
    if config.document_context_path:
        raise ValueError("document descriptions require milvus or legacy mongo_milvus")
    if config.reranker_neighbor_chars:
        raise ValueError("reranker source neighbors require milvus or legacy mongo_milvus")
    if config.candidate_policy != "rrf":
        raise ValueError("candidate quotas require milvus or legacy mongo_milvus")
    return KnowledgeStore(path or config.path, provider=provider, reranker=reranker,
                          contextual=config.contextual_index, parser=config.parser, parse_options=parse_options)


__all__ = ["KnowledgeStore", "create_store"]
