"""Layered session and cross-session memory with local-first persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from .ids import make_id
from .llm import ResearchLLM
from .memory_system import (
    AutonomousMemory,
    CreateFact,
    MemoryDocument,
    MemoryFact,
    MemoryScope,
    MemorySearchResult,
    ScopeType,
    SourceType,
    UpdateFact,
)
from .memory_system.compaction import (
    COMPACTION_SYSTEM_PROMPT,
    CompactionController,
    apply_segment_to_rolling_state,
    authoritative_user_constraints,
    build_compaction_payload,
    current_state_projection,
    estimate_tokens,
    fit_text_values,
    normalize_segment,
    project_memories,
    render_rolling_summary,
    separate_historical_segment_state,
    segment_document_content,
)
from .runtime_config import MemoryRuntimeConfig
from .progress import emit_progress
from .schemas import (
    ConversationMessage,
    MemoryEpisode,
    MemoryPacket,
    ResearchTask,
    SessionMemory,
    utc_now,
)


logger = logging.getLogger(__name__)


class SessionStore(Protocol):
    async def load(self, user_id: str, project_id: str, session_id: str) -> SessionMemory | None: ...

    async def save(self, session: SessionMemory, ttl_seconds: int) -> None: ...

    async def list_sessions(
        self, user_id: str, project_id: str, limit: int = 50, offset: int = 0
    ) -> list[SessionMemory]: ...

    async def close(self) -> None: ...


class LocalMemoryStore:
    """SQLite fallback and durable Episode/Outbox store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._initialize)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    memory_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_scope
                    ON episodes(user_id, project_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS conversation_archive (
                    message_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    run_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_archive_session
                    ON conversation_archive(user_id, project_id, session_id, created_at);
                """
            )

    @staticmethod
    def _key(user_id: str, project_id: str, session_id: str) -> str:
        return f"{user_id}:{project_id}:{session_id}"

    async def load(self, user_id: str, project_id: str, session_id: str) -> SessionMemory | None:
        return await asyncio.to_thread(self._load, user_id, project_id, session_id)

    def _load(self, user_id: str, project_id: str, session_id: str) -> SessionMemory | None:
        key = self._key(user_id, project_id, session_id)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT data, expires_at FROM sessions WHERE memory_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_at"]) < time.time():
                connection.execute("DELETE FROM sessions WHERE memory_key = ?", (key,))
                return None
            return SessionMemory.model_validate_json(row["data"])

    async def save(self, session: SessionMemory, ttl_seconds: int) -> None:
        await asyncio.to_thread(self._save, session, ttl_seconds)

    def _save(self, session: SessionMemory, ttl_seconds: int) -> None:
        key = self._key(session.user_id, session.project_id, session.session_id)
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO sessions(memory_key, data, expires_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(memory_key) DO UPDATE SET
                     data=excluded.data, expires_at=excluded.expires_at,
                     updated_at=excluded.updated_at""",
                (
                    key,
                    session.model_dump_json(),
                    time.time() + ttl_seconds,
                    time.time(),
                ),
            )

    async def list_sessions(
        self, user_id: str, project_id: str, limit: int = 50, offset: int = 0
    ) -> list[SessionMemory]:
        return await asyncio.to_thread(
            self._list_sessions, user_id, project_id, limit, offset
        )

    def _list_sessions(
        self, user_id: str, project_id: str, limit: int, offset: int
    ) -> list[SessionMemory]:
        prefix = f"{user_id}:{project_id}:"
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE expires_at < ?", (now,)
            )
            rows = connection.execute(
                """SELECT data FROM sessions
                   WHERE memory_key LIKE ? ESCAPE '\\'
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (
                    prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",
                    limit,
                    offset,
                ),
            ).fetchall()
        return [SessionMemory.model_validate_json(row["data"]) for row in rows]

    async def save_episode(self, episode: MemoryEpisode) -> None:
        await asyncio.to_thread(self._save_episode, episode)

    def _save_episode(self, episode: MemoryEpisode) -> None:
        data = episode.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO episodes
                   (episode_id, user_id, project_id, created_at, data)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(episode_id) DO UPDATE SET
                     user_id = excluded.user_id,
                     project_id = excluded.project_id,
                     created_at = excluded.created_at,
                     data = excluded.data""",
                (
                    episode.episode_id,
                    episode.user_id,
                    episode.project_id,
                    episode.created_at,
                    data,
                ),
            )

    async def get_episode_by_run(self, run_id: str) -> MemoryEpisode | None:
        return await asyncio.to_thread(self._get_episode_by_run, run_id)

    def _get_episode_by_run(self, run_id: str) -> MemoryEpisode | None:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT data FROM episodes ORDER BY created_at DESC"
            ).fetchall()
        for row in rows:
            episode = MemoryEpisode.model_validate_json(row["data"])
            if episode.run_id == run_id:
                return episode
        return None

    async def archive_message(
        self,
        session: SessionMemory,
        message: ConversationMessage,
        *,
        content: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._archive_message, session, message, content or message.content
        )

    def _archive_message(
        self, session: SessionMemory, message: ConversationMessage, content: str
    ) -> None:
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT content FROM conversation_archive WHERE message_id = ?",
                (message.message_id,),
            ).fetchone()
            if existing:
                if existing["content"] != content:
                    raise ValueError(
                        f"immutable archived message changed: {message.message_id}"
                    )
                return
            connection.execute(
                """INSERT INTO conversation_archive
                   (message_id, user_id, project_id, session_id, turn_id, role,
                    content, run_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.message_id,
                    session.user_id,
                    session.project_id,
                    session.session_id,
                    message.turn_id,
                    message.role,
                    content,
                    message.run_id,
                    message.created_at,
                ),
            )

    async def archived_messages(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._archived_messages, user_id, project_id, session_id, limit, offset
        )

    def _archived_messages(
        self, user_id: str, project_id: str, session_id: str, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM conversation_archive
                   WHERE user_id = ? AND project_id = ? AND session_id = ?
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (user_id, project_id, session_id, limit, offset),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    async def archived_message_count(
        self, user_id: str, project_id: str, session_id: str
    ) -> int:
        return await asyncio.to_thread(
            self._archived_message_count, user_id, project_id, session_id
        )

    def _archived_message_count(
        self, user_id: str, project_id: str, session_id: str
    ) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS message_count FROM conversation_archive
                   WHERE user_id = ? AND project_id = ? AND session_id = ?""",
                (user_id, project_id, session_id),
            ).fetchone()
        return int(row["message_count"] if row else 0)

    async def search_episodes(
        self, user_id: str, project_id: str, query: str, limit: int
    ) -> list[MemoryEpisode]:
        return await asyncio.to_thread(
            self._search_episodes, user_id, project_id, query, limit
        )

    async def list_all_episodes(self, limit: int = 5000) -> list[MemoryEpisode]:
        return await asyncio.to_thread(self._list_all_episodes, limit)

    def _list_all_episodes(self, limit: int) -> list[MemoryEpisode]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT data FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [MemoryEpisode.model_validate_json(row["data"]) for row in rows]

    @staticmethod
    def _terms(value: str) -> set[str]:
        lowered = value.casefold()
        words = set(re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", lowered))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
        words.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
        return {item for item in words if item}

    def _search_episodes(
        self, user_id: str, project_id: str, query: str, limit: int
    ) -> list[MemoryEpisode]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT data FROM episodes WHERE user_id = ? AND project_id = ?
                   ORDER BY created_at DESC LIMIT 200""",
                (user_id, project_id),
            ).fetchall()
        query_terms = self._terms(query)
        scored: list[tuple[int, int, MemoryEpisode]] = []
        for recency, row in enumerate(rows):
            episode = MemoryEpisode.model_validate_json(row["data"])
            haystack = " ".join(
                [episode.research_brief, episode.summary, *episode.open_gaps]
            )
            score = len(query_terms & self._terms(haystack))
            if score or not query_terms:
                scored.append((score, -recency, episode))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    async def close(self) -> None:
        return None


class RedisSessionStore:
    """Redis session store with user/project/session scoped keys."""

    def __init__(self, url: str):
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - depends on optional service
            raise RuntimeError("redis package is required for the Redis session backend") from exc
        self.client = Redis.from_url(url, decode_responses=True)

    @staticmethod
    def _key(user_id: str, project_id: str, session_id: str) -> str:
        return f"stm:session:{user_id}:{project_id}:{session_id}"

    async def load(self, user_id: str, project_id: str, session_id: str) -> SessionMemory | None:
        value = await self.client.get(self._key(user_id, project_id, session_id))
        return SessionMemory.model_validate_json(value) if value else None

    async def save(self, session: SessionMemory, ttl_seconds: int) -> None:
        await self.client.set(
            self._key(session.user_id, session.project_id, session.session_id),
            session.model_dump_json(),
            ex=ttl_seconds,
        )

    async def list_sessions(
        self, user_id: str, project_id: str, limit: int = 50, offset: int = 0
    ) -> list[SessionMemory]:
        pattern = self._key(user_id, project_id, "*")
        keys: list[str] = []
        async for key in self.client.scan_iter(match=pattern, count=100):
            keys.append(key)
        if not keys:
            return []
        values = await self.client.mget(keys)
        sessions = [
            SessionMemory.model_validate_json(value) for value in values if value
        ]
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions[offset : offset + limit]

    async def close(self) -> None:
        await self.client.aclose()


class LayeredMemory:
    def __init__(
        self,
        *,
        config: MemoryRuntimeConfig,
        root: Path,
        llm: ResearchLLM,
        session_store: SessionStore | None = None,
    ):
        self.config = config
        self.root = root
        self.llm = llm
        local_path = Path(config.local_db_path)
        if not local_path.is_absolute():
            local_path = root / local_path
        self.local = LocalMemoryStore(local_path)
        self.session_store = session_store
        self.compaction = CompactionController(
            context_window_tokens=config.context_window_tokens,
            light_ratio=config.compaction_light_ratio,
            query_ratio=config.compaction_query_ratio,
            summary_ratio=config.compaction_summary_ratio,
            emergency_ratio=config.compaction_emergency_ratio,
        )
        self.facts = AutonomousMemory(
            local_path,
            llm=llm,
            embedding_dimensions=config.embedding_dimensions,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_query_prefix=config.embedding_query_prefix,
            embedding_document_prefix=config.embedding_document_prefix,
            embedding_fallback_enabled=config.embedding_fallback_enabled,
            semantic_candidate_limit=config.semantic_candidate_limit,
            lexical_candidate_limit=config.lexical_candidate_limit,
            semantic_dedup_threshold=config.semantic_dedup_threshold,
            minimum_confidence=config.fact_confidence_threshold,
            capture_batch_size=config.capture_batch_size,
            max_facts_per_scope=config.max_facts_per_scope,
            default_valid_days=config.default_fact_valid_days,
        )

    async def open(self) -> None:
        await self.local.open()
        await self.facts.open()
        legacy_episodes = await self.local.list_all_episodes()
        if legacy_episodes:
            await self.facts.upsert_documents(
                [self._episode_document(item) for item in legacy_episodes]
            )
        if self.session_store is None:
            if self.config.session_backend == "redis":
                url = os.getenv(self.config.redis_url_env)
                if not url:
                    raise ValueError(
                        f"{self.config.redis_url_env} is required for session_backend=redis"
                    )
                self.session_store = RedisSessionStore(url)
            else:
                self.session_store = self.local
        logger.info(
            "Memory opened with session_backend=%s project_id=%s",
            self.config.session_backend,
            self.config.project_id,
        )

    async def close(self) -> None:
        if self.session_store is not None and self.session_store is not self.local:
            await self.session_store.close()
        await self.local.close()

    @property
    def ttl_seconds(self) -> int:
        return self.config.session_ttl_hours * 3600

    async def get_session(
        self, user_id: str, session_id: str, project_id: str | None = None
    ) -> SessionMemory:
        assert self.session_store is not None
        project_id = project_id or self.config.project_id
        session = await self.session_store.load(
            user_id, project_id, session_id
        )
        if session is None:
            session = SessionMemory(
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
            )
            await self.session_store.save(session, self.ttl_seconds)
        return session

    async def find_session(
        self, user_id: str, session_id: str, project_id: str | None = None
    ) -> SessionMemory | None:
        """Load an existing session without creating it."""
        assert self.session_store is not None
        return await self.session_store.load(
            user_id, project_id or self.config.project_id, session_id
        )

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 50,
        project_id: str | None = None,
        offset: int = 0,
    ) -> list[SessionMemory]:
        assert self.session_store is not None
        return await self.session_store.list_sessions(
            user_id, project_id or self.config.project_id, limit, offset
        )

    def scope(
        self,
        user_id: str,
        *,
        project_id: str | None = None,
        scope_type: ScopeType = ScopeType.USER,
        agent_name: str | None = None,
    ) -> MemoryScope:
        return MemoryScope(
            scope_type=scope_type,
            user_id=user_id,
            project_id=project_id or self.config.project_id,
            agent_name=agent_name,
        )

    async def create_fact(
        self,
        *,
        user_id: str,
        content: str,
        project_id: str | None = None,
        scope_type: ScopeType = ScopeType.USER,
        category: str = "general",
        expected_valid_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryFact:
        """Create or reinforce a manual fact with full audit history."""

        return await self.facts.create(
            CreateFact(
                scope=self.scope(
                    user_id, project_id=project_id, scope_type=scope_type
                ),
                content=content,
                category=category,
                confidence=1.0,
                source_type=SourceType.MANUAL,
                expected_valid_days=expected_valid_days,
                metadata=metadata or {},
            )
        )

    async def update_fact(
        self,
        fact_id: str,
        *,
        expected_revision: int,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        expected_valid_days: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryFact:
        return await self.facts.update(
            fact_id,
            UpdateFact(
                content=content,
                category=category,
                confidence=confidence,
                expected_valid_days=expected_valid_days,
                metadata=metadata,
            ),
            expected_revision=expected_revision,
        )

    async def delete_fact(
        self, fact_id: str, *, expected_revision: int | None = None
    ) -> MemoryFact:
        return await self.facts.delete(
            fact_id, expected_revision=expected_revision
        )

    async def search_facts(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str | None = None,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        return await self.facts.search(
            query,
            scope=self.scope(user_id, project_id=project_id),
            limit=limit,
        )

    async def set_session_title(
        self, session: SessionMemory, title: str | None
    ) -> SessionMemory:
        normalized = " ".join((title or "").split()).strip()[:80] or None
        updated = session.model_copy(update={"title": normalized})
        await self.save_session(updated)
        return updated

    async def save_session(self, session: SessionMemory) -> None:
        assert self.session_store is not None
        updated = session.model_copy(update={"updated_at": utc_now()})
        await self.session_store.save(updated, self.ttl_seconds)

    async def append_message(
        self,
        session: SessionMemory,
        *,
        role: str,
        content: str,
        turn_id: str,
        run_id: str | None = None,
        archive_content: str | None = None,
    ) -> SessionMemory:
        message = ConversationMessage(
            message_id=make_id("msg", session.session_id, turn_id, role, len(session.messages)),
            turn_id=turn_id,
            role=role,
            content=content,
            run_id=run_id,
        )
        await self.local.archive_message(
            session, message, content=archive_content or content
        )
        updated = session.model_copy(update={"messages": [*session.messages, message]})
        updated = await self._compact_if_needed(updated)
        await self.save_session(updated)
        if role == "assistant" and self.config.automatic_capture_enabled:
            await self.facts.enqueue_turn(
                scope=MemoryScope(
                    scope_type=ScopeType.USER,
                    user_id=updated.user_id,
                    project_id=updated.project_id,
                ),
                thread_id=updated.session_id,
                turn_id=turn_id,
                messages=updated.messages,
            )
        return updated

    async def _compact_if_needed(self, session: SessionMemory) -> SessionMemory:
        approximate_tokens = estimate_tokens(
            "\n".join(item.content for item in session.messages)
        )
        keep = self.config.recent_turns * 2
        if (
            approximate_tokens < self.config.summary_trigger_tokens
            or len(session.messages) <= keep
        ):
            return session
        pressure = self.compaction.assess(approximate_tokens)
        compactable = session.messages[:-keep]
        segment_size = (
            self.config.segment_turns * 2 * pressure.segment_multiplier
        )
        older = compactable[:segment_size]
        payload = build_compaction_payload(
            existing_state=session.rolling_state,
            messages=older,
        )
        try:
            value, _ = await self.llm.complete_json(
                COMPACTION_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=1200,
            )
        except Exception:
            logger.warning("Session compaction degraded; retaining the full message history")
            return session
        segment_id = make_id(
            "segment",
            session.user_id,
            session.project_id,
            session.session_id,
            [item.message_id for item in older],
        )
        try:
            segment = normalize_segment(
                value=value,
                messages=older,
                segment_id=segment_id,
                session_id=session.session_id,
                model_id=str(getattr(self.llm, "model_id", "")),
            )
            segment_metadata = {
                **segment.model_dump(mode="json"),
                "session_id": session.session_id,
                "message_ids": [item.message_id for item in older],
                "start_at": older[0].created_at,
                "end_at": older[-1].created_at,
                # Compatibility aliases consumed by existing retrieval paths.
                "user_corrections": list(segment.corrections),
            }
            await self.facts.upsert_document(
                MemoryDocument(
                    document_id=segment_id,
                    document_type="conversation_segment",
                    scope=self.scope(session.user_id, project_id=session.project_id),
                    content=segment_document_content(segment),
                    metadata=segment_metadata,
                    created_at=older[0].created_at,
                    updated_at=older[-1].created_at,
                ),
                immutable=True,
            )
        except Exception:
            logger.exception(
                "Conversation segment validation/persistence failed; retaining full message history"
            )
            return session
        base_state = session.rolling_state
        if base_state.revision == 0 and session.rolling_summary:
            base_state = base_state.model_copy(
                update={"segment_abstracts": [session.rolling_summary]}
            )
        rolling_state = apply_segment_to_rolling_state(
            base_state,
            segment,
            segment_limit=self.config.rolling_segment_limit,
        )
        rolling_summary = render_rolling_summary(rolling_state)
        logger.info("Compacted session %s", session.session_id)
        return session.model_copy(
            update={
                "messages": session.messages[len(older) :],
                "rolling_summary": rolling_summary,
                "rolling_state": rolling_state,
                "active_entities": list(
                    dict.fromkeys([*session.active_entities, *segment.active_entities])
                )[-20:],
                "user_corrections": list(
                    dict.fromkeys([*session.user_corrections, *segment.corrections])
                )[-30:],
            }
        )

    async def relevant_episodes(
        self, session: SessionMemory, query: str, limit: int
    ) -> list[dict[str, Any]]:
        operation_id = f"memory-retrieval:{session.session_id}"
        emit_progress(
            "memory_retrieval_started",
            phase="memory",
            agent_role="memory",
            status="started",
            message="正在检索会话记忆和长期记忆",
            operation_id=operation_id,
            metrics={"limit": limit},
        )
        candidate_limit = max(limit * 2, limit)
        scope = MemoryScope(
            scope_type=ScopeType.USER,
            user_id=session.user_id,
            project_id=session.project_id,
        )
        project_scope = scope.model_copy(update={"scope_type": ScopeType.PROJECT})
        prepared = (
            await self.facts.prepare_retrieval_query(query)
            if query.strip()
            else None
        )
        # Both searches are read-only when supplied with the same prepared
        # query. Usage accounting is committed after both channels complete.
        fact_matches, document_matches = await asyncio.gather(
            self.facts.search(
                query,
                scope=scope,
                limit=candidate_limit,
                record_usage=False,
                prepared=prepared,
            ),
            self.facts.search_documents(
                query,
                scope=scope,
                document_types={"research_episode", "conversation_segment"},
                limit=candidate_limit,
                prepared=prepared,
            ),
        )
        await self.facts.record_retrieval_usage(
            [match.fact.fact_id for match in fact_matches]
        )
        # Legacy episodes created before document indexing remain searchable
        # during the migration window.
        local = await self.local.search_episodes(
            session.user_id, session.project_id, query, candidate_limit
        )
        mandatory_user_facts = await self.facts.list_facts(
            scope, limit=self.config.mandatory_fact_limit
        )
        mandatory_project_facts = await self.facts.list_facts(
            project_scope, limit=self.config.mandatory_fact_limit
        )
        ranked: list[tuple[float, dict[str, Any]]] = []
        seen_documents: set[str] = set()
        for match in document_matches:
            document = match.document
            seen_documents.add(document.document_id)
            if document.document_type == "research_episode":
                item = {
                    **document.metadata,
                    "origin": "research_episode",
                    "memory_type": "research_episode",
                    "retrieval_score": match.total,
                    "retrieval_channels": match.matched_by,
                }
            else:
                metadata = document.metadata
                item = {
                    "memory_id": document.document_id,
                    "session_id": metadata.get("session_id"),
                    "summary": metadata.get("summary") or document.content,
                    "abstract": metadata.get("abstract") or metadata.get("summary") or "",
                    "entities": metadata.get("active_entities", []),
                    "user_corrections": metadata.get("user_corrections", []),
                    "decisions": metadata.get("decisions", []),
                    "constraints": metadata.get("constraints", []),
                    "open_gaps": metadata.get("unresolved", []),
                    "must_keep_spans": metadata.get("must_keep_spans", []),
                    "detail_ref": metadata.get("detail_ref", {}),
                    "preferences": {},
                    "origin": "conversation_segment",
                    "memory_type": "conversation_segment",
                    "retrieval_score": match.total,
                    "retrieval_channels": match.matched_by,
                }
            ranked.append((match.total, item))
        for rank, item in enumerate(local):
            if item.episode_id in seen_documents:
                continue
            ranked.append(
                (
                    0.7 / (rank + 1),
                    {
                        **item.model_dump(mode="json"),
                        "origin": "research_episode",
                        "memory_type": "research_episode",
                        "retrieval_score": 0.7 / (rank + 1),
                        "retrieval_channels": ["legacy_lexical"],
                    },
                )
            )
        seen_fact_ids: set[str] = set()
        for rank, match in enumerate(fact_matches):
            fact = match.fact
            seen_fact_ids.add(fact.fact_id)
            correction = (
                [fact.content]
                if fact.category in {"correction", "constraint", "project_decision"}
                else []
            )
            preferences = (
                {"remembered_preferences": [fact.content]}
                if fact.category == "preference"
                else {}
            )
            # Blend within-source rank with the fact retrieval score so neither
            # episodes nor facts can monopolize the final top-k list.
            fair_score = 0.55 / (rank + 1) + 0.45 * match.score.total
            ranked.append(
                (
                    fair_score,
                    {
                        "memory_id": fact.fact_id,
                        "summary": fact.content,
                        "category": fact.category,
                        "scope": fact.scope.model_dump(mode="json"),
                        "revision": fact.revision,
                        "confidence": fact.confidence,
                        "user_corrections": correction,
                        "preferences": preferences,
                        "entities": [],
                        "origin": "autonomous_fact",
                        "memory_type": "fact",
                        "retrieval_score": match.score.total,
                        "retrieval_explanation": match.explanation,
                    },
                )
            )
        mandatory_categories = {
            "preference",
            "constraint",
            "correction",
            "project_decision",
        }
        for fact in [*mandatory_user_facts, *mandatory_project_facts]:
            if fact.fact_id in seen_fact_ids or fact.category not in mandatory_categories:
                continue
            mandatory_score = {
                "correction": 0.74,
                "constraint": 0.72,
                "project_decision": 0.70,
                "preference": 0.68,
            }[fact.category]
            ranked.append(
                (
                    mandatory_score,
                    {
                        "memory_id": fact.fact_id,
                        "summary": fact.content,
                        "category": fact.category,
                        "scope": fact.scope.model_dump(mode="json"),
                        "revision": fact.revision,
                        "confidence": fact.confidence,
                        "user_corrections": (
                            [fact.content]
                            if fact.category
                            in {"correction", "constraint", "project_decision"}
                            else []
                        ),
                        "preferences": (
                            {"remembered_preferences": [fact.content]}
                            if fact.category == "preference"
                            else {}
                        ),
                        "entities": [],
                        "origin": "mandatory_fact_reinjection",
                        "memory_type": "fact",
                        "retrieval_score": mandatory_score,
                        "retrieval_explanation": (
                            "deterministic reinjection of active durable state"
                        ),
                    },
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        result = [item for _, item in ranked[:limit]]
        logger.info(
            "Retrieved %d episode/fact memories for session %s",
            len(result),
            session.session_id,
        )
        emit_progress(
            "memory_retrieval_completed",
            phase="memory",
            agent_role="memory",
            status="completed",
            message=f"记忆检索完成，找到 {len(result)} 条相关记录",
            operation_id=operation_id,
            metrics={
                "memory_count": len(result),
                "fact_count": sum(item.get("memory_type") == "fact" for item in result),
                "episode_count": sum(
                    item.get("memory_type") == "research_episode" for item in result
                ),
                "segment_count": sum(
                    item.get("memory_type") == "conversation_segment" for item in result
                ),
            },
        )
        return result

    def _recent_dialogue(self, session: SessionMemory, turns: int | None = None) -> list[dict[str, str]]:
        count = (turns or self.config.recent_turns) * 2
        return [
            {
                "role": item.role,
                "content": item.content[: self.config.hot_message_max_chars],
                "turn_id": item.turn_id,
            }
            for item in session.messages[-count:]
            if item.role in {"user", "assistant"}
        ]

    async def compact_research_message(
        self,
        *,
        run_id: str,
        report_path: str | None,
        full_content: str,
    ) -> str:
        """Keep the full report in the archive and only its projection hot."""

        episode = await self.local.get_episode_by_run(run_id)
        summary = (episode.summary if episode else "").strip()
        if not summary:
            summary = full_content[: self.config.hot_message_max_chars].strip()
        summary = summary[: self.config.hot_message_max_chars]
        parts = [summary, f"Research run: {run_id}"]
        if report_path:
            parts.append(f"完整报告：{report_path}")
        return "\n\n".join(parts)

    async def clarifier_packet(self, session: SessionMemory, query: str) -> MemoryPacket:
        limit = self.config.fact_top_k_clarifier
        candidate_limit = limit * self.config.projection_candidate_multiplier
        candidates = await self.relevant_episodes(
            session, query, candidate_limit
        ) if limit else []
        temporal_candidates = separate_historical_segment_state(
            candidates, session_id=session.session_id
        )
        projection = project_memories(
            temporal_candidates,
            query=query,
            role="clarifier",
            budget_tokens=self.config.clarifier_memory_budget_tokens,
            max_items=limit,
            sentence_limit=self.config.projection_sentence_limit,
            ratios=(
                self.config.compaction_light_ratio,
                self.config.compaction_query_ratio,
                self.config.compaction_summary_ratio,
                self.config.compaction_emergency_ratio,
            ),
        )
        episodes = projection.items
        historical_preferences: dict[str, Any] = {}
        for item in reversed(episodes):
            historical_preferences.update(item.get("preferences", {}))
        historical_preferences.update(session.preferences)
        packet = MemoryPacket(
            role="clarifier",
            query=query,
            session_summary=session.rolling_summary,
            recent_dialogue=self._recent_dialogue(session),
            user_constraints=authoritative_user_constraints(session, episodes),
            preferences=historical_preferences,
            current_state=current_state_projection(session),
            relevant_episodes=episodes,
            historical_segments=[
                item
                for item in episodes
                if item.get("memory_type") == "conversation_segment"
            ],
            source_memory_ids=projection.source_memory_ids,
            projection_level=projection.level,
            budget_tokens=self.config.clarifier_memory_budget_tokens,
        )
        return packet.model_copy(
            update={"estimated_tokens": estimate_tokens(packet.model_dump_json())}
        )

    async def planner_packet(
        self, session: SessionMemory, research_brief: str
    ) -> MemoryPacket:
        limit = self.config.fact_top_k_planner
        candidate_limit = limit * self.config.projection_candidate_multiplier
        candidates = await self.relevant_episodes(
            session, research_brief, candidate_limit
        ) if limit else []
        temporal_candidates = separate_historical_segment_state(
            candidates, session_id=session.session_id
        )
        projection = project_memories(
            temporal_candidates,
            query=research_brief,
            role="planner",
            budget_tokens=self.config.planner_memory_budget_tokens,
            max_items=limit,
            sentence_limit=self.config.projection_sentence_limit,
            ratios=(
                self.config.compaction_light_ratio,
                self.config.compaction_query_ratio,
                self.config.compaction_summary_ratio,
                self.config.compaction_emergency_ratio,
            ),
        )
        episodes = projection.items
        successful: list[str] = []
        failed: list[str] = []
        gaps: list[str] = []
        historical_preferences: dict[str, Any] = {}
        for item in episodes:
            successful.extend(item.get("successful_queries", []))
            failed.extend(item.get("failed_queries", []))
            gaps.extend(item.get("open_gaps", []))
        for item in reversed(episodes):
            historical_preferences.update(item.get("preferences", {}))
        historical_preferences.update(session.preferences)
        packet = MemoryPacket(
            role="planner",
            query=research_brief,
            session_summary=session.rolling_summary,
            recent_dialogue=self._recent_dialogue(session, turns=3),
            user_constraints=authoritative_user_constraints(session, episodes),
            preferences=historical_preferences,
            current_state=current_state_projection(session),
            relevant_episodes=episodes,
            historical_segments=[
                item
                for item in episodes
                if item.get("memory_type") == "conversation_segment"
            ],
            successful_queries=fit_text_values(
                successful,
                query=research_brief,
                limit=20,
                budget_tokens=1200,
            ),
            failed_queries=fit_text_values(
                failed,
                query=research_brief,
                limit=20,
                budget_tokens=1200,
            ),
            open_gaps=fit_text_values(
                [*session.rolling_state.unresolved, *gaps],
                query=research_brief,
                limit=20,
                budget_tokens=1600,
            ),
            source_memory_ids=projection.source_memory_ids,
            projection_level=projection.level,
            budget_tokens=self.config.planner_memory_budget_tokens,
        )
        return packet.model_copy(
            update={"estimated_tokens": estimate_tokens(packet.model_dump_json())}
        )

    @staticmethod
    def worker_packet(
        planner_packet: MemoryPacket,
        task: ResearchTask,
        state: dict[str, Any],
        *,
        budget_tokens: int = 8000,
        sentence_limit: int = 8,
    ) -> MemoryPacket:
        task_query = " ".join(
            [task.objective, task.research_dimension, *task.key_questions]
        )
        query_terms = LocalMemoryStore._terms(task_query)

        def relevant(value: str) -> bool:
            return not query_terms or bool(
                query_terms & LocalMemoryStore._terms(value)
            )

        episode_candidates = [
            item
            for item in planner_packet.relevant_episodes
            if relevant(str(item.get("research_brief") or item.get("summary") or ""))
        ]
        projection = project_memories(
            episode_candidates,
            query=task_query,
            role="worker",
            budget_tokens=budget_tokens,
            max_items=3,
            sentence_limit=sentence_limit,
        )
        episodes = projection.items
        claims = [
            {
                "claim_id": item.get("claim_id"),
                "text": item.get("text"),
                "status": item.get("status"),
            }
            for item in state.get("claims", [])
            if relevant(str(item.get("text") or ""))
        ][-15:]
        packet = MemoryPacket(
            role="worker",
            query=task_query,
            user_constraints=planner_packet.user_constraints,
            preferences=planner_packet.preferences,
            current_state=planner_packet.current_state,
            relevant_episodes=episodes,
            historical_segments=[
                item
                for item in episodes
                if item.get("memory_type") == "conversation_segment"
            ],
            successful_queries=fit_text_values(
                [item for item in planner_packet.successful_queries if relevant(item)],
                query=task_query,
                limit=10,
                budget_tokens=600,
            ),
            failed_queries=fit_text_values(
                [item for item in planner_packet.failed_queries if relevant(item)],
                query=task_query,
                limit=10,
                budget_tokens=600,
            ),
            open_gaps=fit_text_values(
                [item for item in planner_packet.open_gaps if relevant(item)],
                query=task_query,
                limit=10,
                budget_tokens=800,
            ),
            relevant_claims=claims,
            conflicts=[
                str(item.get("missing_information") or "")
                for item in state.get("unresolved_gaps", [])
                if item.get("priority") == "high"
            ][:10],
            source_memory_ids=projection.source_memory_ids,
            projection_level=projection.level,
            budget_tokens=budget_tokens,
        )
        return packet.model_copy(
            update={"estimated_tokens": estimate_tokens(packet.model_dump_json())}
        )

    async def save_episode(self, episode: MemoryEpisode) -> None:
        await self.local.save_episode(episode)
        await self.facts.upsert_document(self._episode_document(episode))

    def _episode_document(self, episode: MemoryEpisode) -> MemoryDocument:
        content = "\n".join(
            item
            for item in [
                episode.research_brief,
                episode.summary,
                "Open gaps: " + "; ".join(episode.open_gaps),
                "Successful queries: " + "; ".join(episode.successful_queries),
                "Failed queries: " + "; ".join(episode.failed_queries),
            ]
            if item.split(":", 1)[-1].strip()
        )
        return MemoryDocument(
            document_id=episode.episode_id,
            document_type="research_episode",
            scope=self.scope(episode.user_id, project_id=episode.project_id),
            content=content,
            metadata=episode.model_dump(mode="json"),
            created_at=episode.created_at,
            updated_at=episode.created_at,
        )

    async def sync_outbox(self) -> None:
        operation_id = "memory-outbox-sync"
        emit_progress(
            "memory_sync_started",
            phase="memory",
            agent_role="memory",
            status="started",
            message="正在抽取、校验并索引长期记忆",
            operation_id=operation_id,
        )
        metrics = await self.facts.process_captures()
        if self.config.lifecycle_enabled:
            lifecycle = await self.facts.maintain()
            metrics.update({f"lifecycle_{key}": value for key, value in lifecycle.items()})
        emit_progress(
            "memory_sync_completed",
            phase="memory",
            agent_role="memory",
            status="completed",
            message=(
                "长期记忆处理完成："
                f"抽取 {metrics['captures_completed']} 批，"
                f"新增 {metrics['facts_created']} 条，"
                f"失败 {metrics['captures_failed']} 批"
            ),
            operation_id=operation_id,
            metrics=metrics,
        )
