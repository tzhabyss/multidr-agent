"""Pluggable local embedding providers with an explicit degraded fallback."""

from __future__ import annotations

import logging
from threading import RLock
from typing import Protocol, Sequence

from .retrieval import hashed_embedding


logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Synchronous provider contract; callers run heavy work in a worker thread."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def degraded(self) -> bool: ...

    def initialize(self) -> None: ...

    def encode_query(self, value: str) -> list[float]: ...

    def encode_documents(self, values: Sequence[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic lexical hash fallback; never presented as semantic search."""

    def __init__(self, dimensions: int = 128):
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"local-hash-v2-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def degraded(self) -> bool:
        return True

    def initialize(self) -> None:
        return None

    def encode_query(self, value: str) -> list[float]:
        return hashed_embedding(value, self._dimensions)

    def encode_documents(self, values: Sequence[str]) -> list[list[float]]:
        return [hashed_embedding(value, self._dimensions) for value in values]


class SentenceTransformerEmbeddingProvider:
    """Normalized multilingual sentence embeddings, loaded lazily on first use."""

    def __init__(
        self,
        model_name: str,
        *,
        dimensions: int = 384,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
    ):
        self.model_name = model_name
        self._dimensions = dimensions
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model = None
        self._lock = RLock()

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def degraded(self) -> bool:
        return False

    def initialize(self) -> None:
        # Fail fast when the optional runtime is absent without downloading the
        # model during service startup. Model weights remain lazy.
        import sentence_transformers  # noqa: F401

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                dimension_getter = getattr(
                    self._model, "get_embedding_dimension", None
                ) or getattr(self._model, "get_sentence_embedding_dimension")
                detected = dimension_getter()
                if detected:
                    self._dimensions = int(detected)
        return self._model

    def _encode(self, values: Sequence[str]) -> list[list[float]]:
        if not values:
            return []
        encoded = self._load().encode(
            list(values),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if getattr(encoded, "ndim", 1) == 1:
            encoded = encoded.reshape(1, -1)
        self._dimensions = int(encoded.shape[1])
        return encoded.astype("float32", copy=False).tolist()

    def encode_query(self, value: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{value}"])[0]

    def encode_documents(self, values: Sequence[str]) -> list[list[float]]:
        return self._encode([f"{self.document_prefix}{value}" for value in values])


class ResilientEmbeddingProvider:
    """Use a real local model when available and mark fallback mode explicitly."""

    def __init__(
        self,
        *,
        provider: str = "sentence_transformers",
        model_name: str = "intfloat/multilingual-e5-small",
        dimensions: int = 384,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        fallback_enabled: bool = True,
    ):
        self.fallback_enabled = fallback_enabled
        self._fallback = HashEmbeddingProvider(dimensions)
        self._active: EmbeddingProvider
        if provider.casefold() in {"hash", "local_hash"}:
            self._active = self._fallback
        elif provider.casefold() in {"sentence_transformers", "sentence-transformers", "auto"}:
            self._active = SentenceTransformerEmbeddingProvider(
                model_name,
                dimensions=dimensions,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
            )
        else:
            raise ValueError(f"unsupported embedding provider: {provider}")
        self._initialized = False
        self._lock = RLock()
        self.last_error: str | None = None

    @property
    def model_id(self) -> str:
        return self._active.model_id

    @property
    def dimensions(self) -> int:
        return self._active.dimensions

    @property
    def degraded(self) -> bool:
        return self._active.degraded

    def _degrade(self, exc: Exception) -> None:
        if not self.fallback_enabled:
            raise exc
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self._active is not self._fallback:
            logger.warning(
                "Semantic embedding unavailable; memory retrieval is using the "
                "deterministic hash fallback: %s",
                self.last_error,
            )
        self._active = self._fallback
        self._active.initialize()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                self._active.initialize()
            except Exception as exc:
                self._degrade(exc)
            self._initialized = True

    def encode_query(self, value: str) -> list[float]:
        self.initialize()
        try:
            return self._active.encode_query(value)
        except Exception as exc:
            with self._lock:
                self._degrade(exc)
            return self._active.encode_query(value)

    def encode_documents(self, values: Sequence[str]) -> list[list[float]]:
        self.initialize()
        try:
            return self._active.encode_documents(values)
        except Exception as exc:
            with self._lock:
                self._degrade(exc)
            return self._active.encode_documents(values)
