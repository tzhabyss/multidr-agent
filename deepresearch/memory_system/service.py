"""Async facade combining capture, CRUD, retrieval, and lifecycle policies."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from ..ids import make_id
from ..llm import ResearchLLM
from .extractor import MemoryExtractor, detect_signals
from .models import (
    ApplyResult,
    CaptureRecord,
    ChangeSet,
    CreateFact,
    FactStatus,
    MemoryDocument,
    MemoryDocumentSearchResult,
    MemoryFact,
    PreparedMemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryStats,
    SearchScore,
    ScopeType,
    UpdateFact,
)
from .repository import MemoryRepository
from .retrieval import (
    lexical_overlap,
    recency_score,
)
from .embeddings import EmbeddingProvider, ResilientEmbeddingProvider
from .vector_index import ScopedVectorIndex


class AutonomousMemory:
    """The project's fully self-hosted Mem0/DeerFlow-inspired memory core."""

    def __init__(
        self,
        path: str | Path,
        *,
        llm: ResearchLLM,
        embedding_dimensions: int = 384,
        embedding_provider: str = "sentence_transformers",
        embedding_model: str = "intfloat/multilingual-e5-small",
        embedding_query_prefix: str = "query: ",
        embedding_document_prefix: str = "passage: ",
        embedding_fallback_enabled: bool = True,
        semantic_candidate_limit: int = 50,
        lexical_candidate_limit: int = 50,
        semantic_dedup_threshold: float = 0.92,
        provider: EmbeddingProvider | None = None,
        minimum_confidence: float = 0.75,
        capture_batch_size: int = 20,
        max_facts_per_scope: int = 500,
        default_valid_days: int = 365,
    ):
        self.repository = MemoryRepository(
            path, embedding_dimensions=embedding_dimensions
        )
        self.extractor = MemoryExtractor(llm)
        self.embedding_provider = provider or ResilientEmbeddingProvider(
            provider=embedding_provider,
            model_name=embedding_model,
            dimensions=embedding_dimensions,
            query_prefix=embedding_query_prefix,
            document_prefix=embedding_document_prefix,
            fallback_enabled=embedding_fallback_enabled,
        )
        self.vector_index = ScopedVectorIndex()
        self.semantic_candidate_limit = semantic_candidate_limit
        self.lexical_candidate_limit = lexical_candidate_limit
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.minimum_confidence = minimum_confidence
        self.capture_batch_size = capture_batch_size
        self.max_facts_per_scope = max_facts_per_scope
        self.default_valid_days = default_valid_days
        self.last_error: str | None = None
        self._index_lock = asyncio.Lock()

    async def open(self) -> None:
        await asyncio.to_thread(self.repository.initialize)
        await asyncio.to_thread(self.embedding_provider.initialize)
        await asyncio.to_thread(
            self.repository.queue_embedding_migration,
            self.embedding_provider.model_id,
        )
        await self._process_indexes()

    async def _process_indexes(self) -> tuple[int, int, int, int]:
        """Drain both projections and invalidate exact-index caches on writes."""

        async with self._index_lock:
            totals = [0, 0, 0, 0]
            # A lazy provider can switch once to the deterministic fallback.
            # Run one convergence pass so read-your-writes still holds for the
            # final model instead of leaving the migrated rows for a later read.
            for _attempt in range(2):
                original_model = self.embedding_provider.model_id
                indexed, failed = await asyncio.to_thread(
                    self.repository.process_index_outbox, self.embedding_provider
                )
                document_indexed, document_failed = await asyncio.to_thread(
                    self.repository.process_document_index_outbox,
                    self.embedding_provider,
                )
                totals[0] += indexed
                totals[1] += failed
                totals[2] += document_indexed
                totals[3] += document_failed
                if indexed or document_indexed:
                    self.vector_index.invalidate()
                if self.embedding_provider.model_id == original_model:
                    break
                await asyncio.to_thread(
                    self.repository.queue_embedding_migration,
                    self.embedding_provider.model_id,
                )
            return totals[0], totals[1], totals[2], totals[3]

    async def prepare_retrieval_query(self, query: str) -> PreparedMemoryQuery:
        """Build a read-only query context reusable across retrieval channels."""

        normalized = query.strip()
        if not normalized:
            raise ValueError("memory retrieval query must not be blank")
        # Direct CRUD paths drain their indexes before returning. This drain
        # also covers capture/outbox writes and preserves existing consistency.
        await self._process_indexes()
        before_model = self.embedding_provider.model_id
        query_vector = await asyncio.to_thread(
            self.embedding_provider.encode_query, query
        )
        if self.embedding_provider.model_id != before_model:
            await asyncio.to_thread(
                self.repository.queue_embedding_migration,
                self.embedding_provider.model_id,
            )
            await self._process_indexes()
        return PreparedMemoryQuery(
            query=query,
            vector=query_vector,
            embedding_model=self.embedding_provider.model_id,
            dimensions=len(query_vector),
            degraded=self.embedding_provider.degraded,
        )

    @staticmethod
    def _validate_prepared_query(
        query: str, prepared: PreparedMemoryQuery
    ) -> None:
        if prepared.query != query:
            raise ValueError("prepared memory query does not match query text")

    async def record_retrieval_usage(self, fact_ids: list[str]) -> None:
        """Record selected facts after a read-only multi-channel retrieval."""

        await asyncio.to_thread(self.repository.record_usage, fact_ids)

    async def create(self, command: CreateFact) -> MemoryFact:
        fact = await asyncio.to_thread(self.repository.create, command)
        await self._process_indexes()
        return fact

    async def get(self, fact_id: str) -> MemoryFact | None:
        return await asyncio.to_thread(self.repository.get, fact_id)

    async def update(
        self, fact_id: str, command: UpdateFact, *, expected_revision: int
    ) -> MemoryFact:
        fact = await asyncio.to_thread(
            self.repository.update,
            fact_id,
            command,
            expected_revision=expected_revision,
        )
        await self._process_indexes()
        return fact

    async def delete(
        self, fact_id: str, *, expected_revision: int | None = None
    ) -> MemoryFact:
        fact = await asyncio.to_thread(
            self.repository.delete,
            fact_id,
            expected_revision=expected_revision,
        )
        await self._process_indexes()
        return fact

    async def reinforce(
        self,
        fact_id: str,
        *,
        confidence: float = 0.8,
        evidence_message_ids: list[str] | None = None,
    ) -> MemoryFact:
        return await asyncio.to_thread(
            self.repository.reinforce,
            fact_id,
            confidence=confidence,
            evidence_message_ids=evidence_message_ids,
        )

    async def history(self, fact_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.repository.history, fact_id, limit)

    async def list_facts(
        self,
        scope: MemoryScope,
        *,
        status: FactStatus | None = FactStatus.ACTIVE,
        limit: int = 100,
    ) -> list[MemoryFact]:
        return await asyncio.to_thread(
            self.repository.list_facts, scope, status=status, limit=limit
        )

    async def enqueue_turn(
        self,
        *,
        scope: MemoryScope,
        thread_id: str,
        turn_id: str,
        messages: list[Any],
    ) -> bool:
        selected: list[dict[str, str]] = []
        for item in messages:
            item_turn = getattr(item, "turn_id", None) or (
                item.get("turn_id") if isinstance(item, dict) else None
            )
            if item_turn != turn_id:
                continue
            role = getattr(item, "role", None) or item.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = getattr(item, "content", None) or item.get("content", "")
            message_id = getattr(item, "message_id", None) or item.get("message_id", "")
            # Assistant output is context only and intentionally truncated.
            maximum = 4000 if role == "assistant" else 8000
            selected.append(
                {
                    "message_id": str(message_id),
                    "role": str(role),
                    "content": str(content)[:maximum],
                }
            )
        if not any(item["role"] == "user" for item in selected):
            return False
        capture = CaptureRecord(
            capture_id=make_id(
                "capture", scope.key, thread_id, turn_id, [item["message_id"] for item in selected]
            ),
            scope=scope,
            thread_id=thread_id,
            turn_id=turn_id,
            messages=selected,
            signals=detect_signals(selected),
        )
        return await asyncio.to_thread(self.repository.enqueue_capture, capture)

    async def process_captures(self) -> dict[str, int]:
        completed = failed = created = reinforced = capacity_trimmed = 0
        examined: set[str] = set()
        while completed + failed < self.capture_batch_size:
            pending = await asyncio.to_thread(
                self.repository.pending_captures,
                self.capture_batch_size - completed - failed,
            )
            pending = [item for item in pending if item.capture_id not in examined]
            if not pending:
                break
            for capture in pending:
                examined.add(capture.capture_id)
                try:
                    user_text = " ".join(
                        item.get("content", "")
                        for item in capture.messages
                        if item.get("role") == "user"
                    )
                    existing_matches = await self.search(
                        user_text,
                        scope=capture.scope,
                        limit=20,
                        record_usage=False,
                    )
                    existing = [item.fact for item in existing_matches]
                    changeset = await self.extractor.extract(capture, existing)
                    changeset = await self._guard_semantic_duplicates(
                        capture.scope, changeset
                    )
                    result: ApplyResult = await asyncio.to_thread(
                        self.repository.apply_changeset,
                        capture,
                        changeset,
                        minimum_confidence=self.minimum_confidence,
                    )
                except Exception as exc:
                    failed += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    await asyncio.to_thread(
                        self.repository.mark_capture_failed,
                        capture.capture_id,
                        self.last_error,
                    )
                    continue
                completed += 1
                created += len(result.created)
                reinforced += len(result.reinforced)
                self.last_error = None
                for scope_type in {candidate.scope for candidate in changeset.candidates}:
                    target_scope = capture.scope.model_copy(
                        update={"scope_type": ScopeType(scope_type)}
                    )
                    try:
                        capacity_trimmed += await asyncio.to_thread(
                            self.repository.enforce_capacity,
                            target_scope,
                            self.max_facts_per_scope,
                        )
                    except Exception as exc:
                        # Admission already committed atomically.  Lifecycle
                        # maintenance is best-effort and must not requeue it.
                        self.last_error = f"{type(exc).__name__}: {exc}"
        indexed, index_failed, documents_indexed, document_indexes_failed = (
            await self._process_indexes()
        )
        return {
            "captures_completed": completed,
            "captures_failed": failed,
            "facts_created": created,
            "facts_reinforced": reinforced,
            "capacity_trimmed": capacity_trimmed,
            "facts_indexed": indexed,
            "indexes_failed": index_failed,
            "documents_indexed": documents_indexed,
            "document_indexes_failed": document_indexes_failed,
        }

    @staticmethod
    def _relation_guard_compatible(left: str, right: str) -> bool:
        """Do not collapse facts that differ in numbers, dates, or polarity."""

        number_pattern = r"(?<![\w.])\d+(?:[.,]\d+)?%?"
        left_numbers = set(re.findall(number_pattern, left))
        right_numbers = set(re.findall(number_pattern, right))
        if (left_numbers or right_numbers) and left_numbers != right_numbers:
            return False
        negations = {
            "不", "没", "没有", "禁止", "不要", "从不", "并非",
            "not", "no", "never", "without", "forbid", "forbidden",
        }
        left_terms = set(re.findall(r"[a-z]+|[\u4e00-\u9fff]{1,2}", left.casefold()))
        right_terms = set(re.findall(r"[a-z]+|[\u4e00-\u9fff]{1,2}", right.casefold()))
        left_negative = bool(left_terms & negations) or any(item in left for item in negations)
        right_negative = bool(right_terms & negations) or any(item in right for item in negations)
        return left_negative == right_negative

    async def _guard_semantic_duplicates(
        self, scope: MemoryScope, changeset: ChangeSet
    ) -> ChangeSet:
        """Mem0-style relation admission backed by model similarity and hard guards."""

        if self.embedding_provider.degraded:
            return changeset
        guarded = []
        for candidate in changeset.candidates:
            if candidate.relation in {"duplicate", "reinforce"} and candidate.target_id:
                target = await self.get(candidate.target_id)
                if target is not None and not self._relation_guard_compatible(
                    candidate.content, target.content
                ):
                    candidate = candidate.model_copy(
                        update={"relation": "new", "target_id": None}
                    )
            if candidate.relation != "new":
                guarded.append(candidate)
                continue
            matches = await self.search(
                candidate.content, scope=scope, limit=5, record_usage=False
            )
            nearest = max(matches, key=lambda item: item.score.semantic, default=None)
            if (
                not self.embedding_provider.degraded
                and nearest is not None
                and nearest.score.semantic >= self.semantic_dedup_threshold
                and nearest.fact.category == candidate.category
                and self._relation_guard_compatible(
                    candidate.content, nearest.fact.content
                )
            ):
                candidate = candidate.model_copy(
                    update={"relation": "duplicate", "target_id": nearest.fact.fact_id}
                )
            guarded.append(candidate)
        return changeset.model_copy(update={"candidates": guarded})

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 5,
        record_usage: bool = True,
        prepared: PreparedMemoryQuery | None = None,
    ) -> list[MemorySearchResult]:
        if limit <= 0 or not query.strip():
            return []
        context = prepared or await self.prepare_retrieval_query(query)
        self._validate_prepared_query(query, context)
        vector_rows = await asyncio.to_thread(
            self.repository.fact_vector_rows,
            scope,
            model=context.embedding_model,
            dimensions=context.dimensions,
        )
        signature = await asyncio.to_thread(
            self.repository.scope_signature, scope
        )
        scope_key = "|".join(self.repository.authorized_scope_keys(scope))
        semantic_matches = await asyncio.to_thread(
            self.vector_index.search,
            cache_key=(
                f"fact:{scope_key}:{context.embedding_model}:"
                f"{context.dimensions}"
            ),
            signature=signature,
            rows=vector_rows,
            query_vector=context.vector,
            limit=self.semantic_candidate_limit,
        )
        semantic_scores = dict(semantic_matches)
        rows = await asyncio.to_thread(
            self.repository.retrieval_rows,
            scope,
            query,
            limit=self.semantic_candidate_limit + self.lexical_candidate_limit,
            candidate_ids=[item[0] for item in semantic_matches],
            lexical_limit=self.lexical_candidate_limit,
        )
        scored: list[MemorySearchResult] = []
        minimum_semantic = 0.18 if context.degraded else 0.25
        for fact, _packed, _dimensions, fts_rank in rows:
            overlap = lexical_overlap(query, fact.content)
            lexical = max(overlap, fts_rank)
            semantic = semantic_scores.get(fact.fact_id, 0.0)
            # Avoid injecting unrelated facts merely because confidence/recency is high.
            if lexical <= 0 and semantic < minimum_semantic:
                continue
            recency = recency_score(fact.updated_at)
            reinforcement = min(1.0, fact.confirmation_count / 5)
            total = min(
                1.0,
                0.35 * lexical
                + 0.35 * semantic
                + 0.15 * fact.confidence
                + 0.10 * recency
                + 0.05 * reinforcement,
            )
            matched = []
            if fts_rank > 0:
                matched.append("fts5")
            if overlap > 0:
                matched.append("lexical_overlap")
            if semantic >= minimum_semantic:
                matched.append(
                    "hash_vector_fallback"
                    if context.degraded
                    else "semantic_vector"
                )
            score = SearchScore(
                lexical=lexical,
                semantic=semantic,
                confidence=fact.confidence,
                recency=recency,
                reinforcement=reinforcement,
                total=total,
            )
            scored.append(
                MemorySearchResult(
                    fact=fact,
                    score=score,
                    matched_by=matched,
                    explanation=(
                        f"hybrid score={total:.3f}; lexical={lexical:.3f}; "
                        f"vector={semantic:.3f}; confidence={fact.confidence:.3f}; "
                        f"confirmations={fact.confirmation_count}"
                    ),
                )
            )
        scored.sort(key=lambda item: item.score.total, reverse=True)
        selected = scored[:limit]
        if record_usage:
            await asyncio.to_thread(
                self.repository.record_usage,
                [item.fact.fact_id for item in selected],
            )
        return selected

    async def upsert_document(
        self, document: MemoryDocument, *, immutable: bool = False
    ) -> bool:
        changed = await asyncio.to_thread(
            self.repository.upsert_document, document, immutable=immutable
        )
        if changed:
            await self._process_indexes()
        return changed

    async def upsert_documents(self, documents: list[MemoryDocument]) -> int:
        changed = 0
        for document in documents:
            changed += int(
                await asyncio.to_thread(self.repository.upsert_document, document)
            )
        if changed:
            await self._process_indexes()
        return changed

    async def search_documents(
        self,
        query: str,
        *,
        scope: MemoryScope,
        document_types: set[str] | None = None,
        limit: int = 5,
        prepared: PreparedMemoryQuery | None = None,
    ) -> list[MemoryDocumentSearchResult]:
        if limit <= 0 or not query.strip():
            return []
        context = prepared or await self.prepare_retrieval_query(query)
        self._validate_prepared_query(query, context)
        rows = await asyncio.to_thread(
            self.repository.document_vector_rows,
            scope,
            model=context.embedding_model,
            dimensions=context.dimensions,
            document_types=document_types,
        )
        signature = await asyncio.to_thread(
            self.repository.scope_signature, scope, document=True
        )
        scope_key = "|".join(self.repository.authorized_scope_keys(scope))
        type_key = ",".join(sorted(document_types or {"*"}))
        semantic_matches = await asyncio.to_thread(
            self.vector_index.search,
            cache_key=(
                f"document:{scope_key}:{type_key}:"
                f"{context.embedding_model}:{context.dimensions}"
            ),
            signature=signature,
            rows=rows,
            query_vector=context.vector,
            limit=self.semantic_candidate_limit,
        )
        semantic_scores = dict(semantic_matches)
        candidates = await asyncio.to_thread(
            self.repository.document_candidates,
            scope,
            query,
            candidate_ids=[item[0] for item in semantic_matches],
            document_types=document_types,
            lexical_limit=self.lexical_candidate_limit,
            limit=self.semantic_candidate_limit + self.lexical_candidate_limit,
        )
        minimum_semantic = 0.18 if context.degraded else 0.25
        results: list[MemoryDocumentSearchResult] = []
        for document, fts_rank in candidates:
            lexical = max(fts_rank, lexical_overlap(query, document.content))
            semantic = semantic_scores.get(document.document_id, 0.0)
            if lexical <= 0 and semantic < minimum_semantic:
                continue
            recency = recency_score(document.updated_at)
            total = min(1.0, 0.40 * semantic + 0.40 * lexical + 0.20 * recency)
            matched: list[str] = []
            if fts_rank > 0:
                matched.append("fts5")
            if lexical > fts_rank:
                matched.append("lexical_overlap")
            if semantic >= minimum_semantic:
                matched.append(
                    "hash_vector_fallback"
                    if context.degraded
                    else "semantic_vector"
                )
            results.append(
                MemoryDocumentSearchResult(
                    document=document,
                    lexical=lexical,
                    semantic=semantic,
                    recency=recency,
                    total=total,
                    matched_by=matched,
                )
            )
        results.sort(key=lambda item: item.total, reverse=True)
        return results[:limit]

    async def maintain(self, scope: MemoryScope | None = None) -> dict[str, int]:
        expired = await asyncio.to_thread(
            self.repository.expire_stale,
            default_valid_days=self.default_valid_days,
        )
        trimmed = 0
        if scope is not None:
            trimmed = await asyncio.to_thread(
                self.repository.enforce_capacity, scope, self.max_facts_per_scope
            )
        indexed, failed, document_indexed, document_failed = await self._process_indexes()
        return {
            "expired": expired,
            "capacity_trimmed": trimmed,
            "indexed": indexed,
            "index_failed": failed,
            "documents_indexed": document_indexed,
            "document_index_failed": document_failed,
        }

    async def stats(self, scope: MemoryScope | None = None) -> MemoryStats:
        stats = await asyncio.to_thread(self.repository.stats, scope)
        return stats.model_copy(
            update={
                "embedding_model": self.embedding_provider.model_id,
                "embedding_dimensions": self.embedding_provider.dimensions,
                "embedding_degraded": self.embedding_provider.degraded,
                "embedding_error": getattr(self.embedding_provider, "last_error", None),
            }
        )
