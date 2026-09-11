"""Small per-scope exact vector index with NumPy acceleration and caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .retrieval import cosine, unpack_vector


@dataclass(slots=True)
class _CacheEntry:
    signature: str
    identifiers: list[str]
    matrix: object


class ScopedVectorIndex:
    """Exact cosine search is sufficient until a scope reaches ANN scale."""

    def __init__(self):
        self._cache: dict[str, _CacheEntry] = {}

    def invalidate(self, prefix: str | None = None) -> None:
        if prefix is None:
            self._cache.clear()
            return
        for key in [item for item in self._cache if item.startswith(prefix)]:
            self._cache.pop(key, None)

    def search(
        self,
        *,
        cache_key: str,
        signature: str,
        rows: Iterable[tuple[str, bytes, int]],
        query_vector: list[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        if limit <= 0 or not query_vector:
            return []
        entry = self._cache.get(cache_key)
        if entry is None or entry.signature != signature:
            identifiers: list[str] = []
            vectors: list[list[float]] = []
            expected = len(query_vector)
            for identifier, packed, dimensions in rows:
                if int(dimensions) != expected:
                    continue
                vector = unpack_vector(packed, expected)
                if vector:
                    identifiers.append(identifier)
                    vectors.append(vector)
            try:
                import numpy as np

                matrix: object = np.asarray(vectors, dtype=np.float32)
            except ImportError:  # deterministic dependency-free fallback
                matrix = vectors
            entry = _CacheEntry(signature, identifiers, matrix)
            self._cache[cache_key] = entry
        if not entry.identifiers:
            return []
        try:
            import numpy as np

            matrix = entry.matrix
            if not isinstance(matrix, np.ndarray):
                raise TypeError
            scores = matrix @ np.asarray(query_vector, dtype=np.float32)
            count = min(limit, len(entry.identifiers))
            if count == len(entry.identifiers):
                indexes = np.argsort(scores)[::-1]
            else:
                indexes = np.argpartition(scores, -count)[-count:]
                indexes = indexes[np.argsort(scores[indexes])[::-1]]
            return [
                (entry.identifiers[int(index)], max(0.0, min(1.0, float(scores[index]))))
                for index in indexes
            ]
        except (ImportError, TypeError):
            scores = [
                (identifier, cosine(query_vector, vector))
                for identifier, vector in zip(
                    entry.identifiers, entry.matrix, strict=True  # type: ignore[arg-type]
                )
            ]
            scores.sort(key=lambda item: item[1], reverse=True)
            return scores[:limit]
