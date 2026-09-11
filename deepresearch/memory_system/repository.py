"""Transactional SQLite repository for autonomous long-term memory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from ..schemas import utc_now
from .models import (
    ApplyResult,
    CaptureRecord,
    ChangeSet,
    CreateFact,
    FactCandidate,
    FactStatus,
    MemoryDocument,
    MemoryFact,
    MemoryScope,
    MemoryStats,
    ScopeType,
    SourceType,
    UpdateFact,
)
from .retrieval import (
    content_hash,
    fts_query_expression,
    lexical_index_text,
    pack_vector,
)


class RevisionConflictError(RuntimeError):
    """Raised when optimistic concurrency detects a stale writer."""


class FactNotFoundError(LookupError):
    pass


class MemoryRepository:
    def __init__(self, path: str | Path, *, embedding_dimensions: int = 128):
        self.path = Path(path)
        self.fts_enabled = False
        self.document_fts_enabled = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _retry_at(attempts: int) -> str:
        delay = min(300, 2 ** min(8, max(0, attempts)))
        return (datetime.fromisoformat(utc_now()) + timedelta(seconds=delay)).isoformat()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def write_connection(self) -> Iterator[sqlite3.Connection]:
        """Serialize read-before-write transactions across SQLite connections.

        ``with connection`` starts a deferred transaction, so concurrent
        callers can all observe a missing idempotency key before any of them
        obtains the write lock. ``BEGIN IMMEDIATE`` acquires that lock before
        the lookup and makes the operation reservation plus state transition a
        single atomic critical section.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _ensure_fts_v2(
        connection: sqlite3.Connection,
        *,
        table: str,
        identifier: str,
        kind_column: str,
        source_table: str,
        outbox_table: str,
        active_only: bool,
    ) -> bool:
        """Migrate a derived FTS projection to the symmetric lexical schema.

        The source tables remain authoritative.  Migration drops only an old
        derived FTS projection and atomically queues every missing source row
        for reconstruction through the existing index outbox.
        """

        definition = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
        ).fetchone()
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        } if definition else set()
        sql = str(definition["sql"] or "").upper() if definition else ""
        is_v2 = (
            "VIRTUAL TABLE" in sql
            and "FTS5" in sql
            and "search_text" in columns
        )
        if definition and not is_v2:
            connection.execute(f"DROP TABLE {table}")
        try:
            connection.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5(
                       {identifier} UNINDEXED,
                       content UNINDEXED,
                       {kind_column} UNINDEXED,
                       search_text
                   )"""
            )
            definition = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()
            enabled = bool(
                definition
                and "VIRTUAL TABLE" in str(definition["sql"] or "").upper()
                and "FTS5" in str(definition["sql"] or "").upper()
            )
        except sqlite3.OperationalError:
            connection.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                       {identifier} TEXT PRIMARY KEY,
                       content TEXT NOT NULL,
                       {kind_column} TEXT NOT NULL,
                       search_text TEXT NOT NULL
                   )"""
            )
            enabled = False

        now = utc_now()
        status_clause = "AND source.status = 'active'" if active_only else ""
        connection.execute(
            f"""INSERT INTO {outbox_table}
                   ({identifier}, attempts, available_at, created_at)
                SELECT source.{identifier}, 0, ?, ?
                FROM {source_table} source
                LEFT JOIN {table} projection
                  ON projection.{identifier} = source.{identifier}
                WHERE projection.{identifier} IS NULL {status_clause}
                ON CONFLICT({identifier}) DO NOTHING""",
            (now, now),
        )
        return enabled

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_manifests (
                    scope_key TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_facts (
                    fact_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_name TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_episode_id TEXT,
                    source_thread_id TEXT,
                    evidence_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    expected_valid_days INTEGER,
                    last_confirmed_at TEXT NOT NULL,
                    confirmation_count INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT,
                    consolidated_from_json TEXT NOT NULL DEFAULT '[]',
                    revision INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(superseded_by) REFERENCES memory_facts(fact_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_facts_scope_status
                    ON memory_facts(user_id, project_id, scope_type, status, updated_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_facts_active_hash
                    ON memory_facts(scope_key, content_hash) WHERE status = 'active';

                CREATE TABLE IF NOT EXISTS memory_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    operation_id TEXT,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_history_fact
                    ON memory_history(fact_id, revision DESC);

                CREATE TABLE IF NOT EXISTS memory_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT,
                    action TEXT NOT NULL,
                    fact_id TEXT,
                    scope_key TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    fact_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    FOREIGN KEY(fact_id) REFERENCES memory_facts(fact_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_index_outbox (
                    fact_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(fact_id) REFERENCES memory_facts(fact_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_capture_outbox (
                    capture_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_name TEXT,
                    scope_type TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    signals_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memory_capture_pending
                    ON memory_capture_outbox(status, available_at, created_at);

                CREATE TABLE IF NOT EXISTS memory_watermarks (
                    scope_key TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    last_capture_id TEXT NOT NULL,
                    last_turn_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(scope_key, thread_id)
                );

                CREATE TABLE IF NOT EXISTS memory_applied_operations (
                    operation_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_fact_usage (
                    fact_id TEXT PRIMARY KEY,
                    retrieval_count INTEGER NOT NULL DEFAULT 0,
                    last_retrieved_at TEXT,
                    FOREIGN KEY(fact_id) REFERENCES memory_facts(fact_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_documents (
                    document_id TEXT PRIMARY KEY,
                    document_type TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_name TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_documents_scope_type
                    ON memory_documents(scope_key, document_type, updated_at DESC);

                CREATE TABLE IF NOT EXISTS memory_document_embeddings (
                    document_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES memory_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_document_index_outbox (
                    document_id TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES memory_documents(document_id) ON DELETE CASCADE
                );
                """
            )
            self.fts_enabled = self._ensure_fts_v2(
                connection,
                table="memory_fts",
                identifier="fact_id",
                kind_column="category",
                source_table="memory_facts",
                outbox_table="memory_index_outbox",
                active_only=True,
            )
            self.document_fts_enabled = self._ensure_fts_v2(
                connection,
                table="memory_document_fts",
                identifier="document_id",
                kind_column="document_type",
                source_table="memory_documents",
                outbox_table="memory_document_index_outbox",
                active_only=False,
            )

    @staticmethod
    def _scope_from_row(row: sqlite3.Row) -> MemoryScope:
        return MemoryScope(
            scope_type=ScopeType(row["scope_type"]),
            user_id=row["user_id"],
            project_id=row["project_id"],
            agent_name=row["agent_name"],
        )

    @classmethod
    def _fact_from_row(cls, row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            fact_id=row["fact_id"],
            scope=cls._scope_from_row(row),
            content=row["content"],
            content_hash=row["content_hash"],
            category=row["category"],
            confidence=float(row["confidence"]),
            status=FactStatus(row["status"]),
            source_type=SourceType(row["source_type"]),
            source_episode_id=row["source_episode_id"],
            source_thread_id=row["source_thread_id"],
            evidence_message_ids=json.loads(row["evidence_message_ids_json"]),
            expected_valid_days=row["expected_valid_days"],
            last_confirmed_at=row["last_confirmed_at"],
            confirmation_count=int(row["confirmation_count"]),
            superseded_by=row["superseded_by"],
            consolidated_from=json.loads(row["consolidated_from_json"]),
            revision=int(row["revision"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @classmethod
    def _document_from_row(cls, row: sqlite3.Row) -> MemoryDocument:
        return MemoryDocument(
            document_id=row["document_id"],
            document_type=row["document_type"],
            scope=cls._scope_from_row(row),
            content=row["content"],
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def authorized_scope_keys(scope: MemoryScope) -> list[str]:
        keys = [
            scope.model_copy(update={"scope_type": ScopeType.USER}).key,
            scope.model_copy(update={"scope_type": ScopeType.PROJECT}).key,
        ]
        if scope.scope_type == ScopeType.AGENT:
            keys.append(scope.key)
        return list(dict.fromkeys(keys))

    def scope_signature(self, scope: MemoryScope, *, document: bool = False) -> str:
        scope_keys = self.authorized_scope_keys(scope)
        placeholders = ", ".join("?" for _ in scope_keys)
        with self.connection() as connection:
            if document:
                row = connection.execute(
                    f"""SELECT COUNT(*) AS amount, COALESCE(MAX(updated_at), '') AS latest
                        FROM memory_documents WHERE scope_key IN ({placeholders})""",
                    scope_keys,
                ).fetchone()
                indexed = connection.execute(
                    f"""SELECT COUNT(*) AS amount, COALESCE(MAX(e.indexed_at), '') AS latest
                        FROM memory_document_embeddings e
                        JOIN memory_documents d ON d.document_id = e.document_id
                        WHERE d.scope_key IN ({placeholders})""",
                    scope_keys,
                ).fetchone()
            else:
                row = connection.execute(
                    f"""SELECT COALESCE(SUM(revision), 0) AS amount,
                               COALESCE(MAX(updated_at), '') AS latest
                        FROM memory_manifests WHERE scope_key IN ({placeholders})""",
                    scope_keys,
                ).fetchone()
                indexed = connection.execute(
                    f"""SELECT COUNT(*) AS amount, COALESCE(MAX(e.indexed_at), '') AS latest
                        FROM memory_embeddings e JOIN memory_facts f ON f.fact_id = e.fact_id
                        WHERE f.scope_key IN ({placeholders}) AND f.status = 'active'""",
                    scope_keys,
                ).fetchone()
        return f"{row['amount']}:{row['latest']}:{indexed['amount']}:{indexed['latest']}"

    @staticmethod
    def _snapshot(fact: MemoryFact) -> str:
        return fact.model_dump_json()

    def _touch_manifest(self, connection: sqlite3.Connection, scope: MemoryScope) -> None:
        connection.execute(
            """INSERT INTO memory_manifests(scope_key, revision, updated_at)
               VALUES (?, 1, ?)
               ON CONFLICT(scope_key) DO UPDATE SET
                 revision = revision + 1, updated_at = excluded.updated_at""",
            (scope.key, utc_now()),
        )

    def _record_history(
        self,
        connection: sqlite3.Connection,
        fact: MemoryFact,
        event_type: str,
        *,
        operation_id: str | None,
        actor: str,
    ) -> None:
        connection.execute(
            """INSERT INTO memory_history
               (fact_id, scope_key, revision, event_type, snapshot_json,
                operation_id, actor, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.fact_id,
                fact.scope.key,
                fact.revision,
                event_type,
                self._snapshot(fact),
                operation_id,
                actor,
                utc_now(),
            ),
        )
        connection.execute(
            """INSERT INTO memory_audit
               (operation_id, action, fact_id, scope_key, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (operation_id, event_type, fact.fact_id, fact.scope.key, "{}", utc_now()),
        )

    @staticmethod
    def _queue_index(connection: sqlite3.Connection, fact_id: str) -> None:
        now = utc_now()
        connection.execute(
            """INSERT INTO memory_index_outbox(fact_id, attempts, available_at, created_at)
               VALUES (?, 0, ?, ?)
               ON CONFLICT(fact_id) DO UPDATE SET
                 attempts = 0, last_error = NULL, available_at = excluded.available_at,
                 created_at = excluded.created_at""",
            (fact_id, now, now),
        )

    def _insert_fact(
        self,
        connection: sqlite3.Connection,
        command: CreateFact,
        *,
        operation_id: str | None,
        actor: str,
    ) -> MemoryFact:
        now = utc_now()
        normalized = " ".join(command.content.split()).strip()
        fact = MemoryFact(
            fact_id=f"fact_{uuid.uuid4().hex}",
            scope=command.scope,
            content=normalized,
            content_hash=content_hash(normalized),
            category=command.category.strip() or "general",
            confidence=command.confidence,
            source_type=command.source_type,
            source_episode_id=command.source_episode_id,
            source_thread_id=command.source_thread_id,
            evidence_message_ids=list(dict.fromkeys(command.evidence_message_ids)),
            expected_valid_days=command.expected_valid_days,
            metadata=command.metadata,
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
        )
        connection.execute(
            """INSERT INTO memory_facts
               (fact_id, scope_key, scope_type, user_id, project_id, agent_name,
                content, content_hash, category, confidence, status, source_type,
                source_episode_id, source_thread_id, evidence_message_ids_json,
                expected_valid_days, last_confirmed_at, confirmation_count,
                superseded_by, consolidated_from_json, revision, metadata_json,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.fact_id,
                fact.scope.key,
                fact.scope.scope_type.value,
                fact.scope.user_id,
                fact.scope.project_id,
                fact.scope.agent_name,
                fact.content,
                fact.content_hash,
                fact.category,
                fact.confidence,
                fact.status.value,
                fact.source_type.value,
                fact.source_episode_id,
                fact.source_thread_id,
                json.dumps(fact.evidence_message_ids, ensure_ascii=False),
                fact.expected_valid_days,
                fact.last_confirmed_at,
                fact.confirmation_count,
                fact.superseded_by,
                json.dumps(fact.consolidated_from, ensure_ascii=False),
                fact.revision,
                json.dumps(fact.metadata, ensure_ascii=False),
                fact.created_at,
                fact.updated_at,
            ),
        )
        self._touch_manifest(connection, fact.scope)
        self._record_history(
            connection, fact, "created", operation_id=operation_id, actor=actor
        )
        self._queue_index(connection, fact.fact_id)
        return fact

    def create(
        self,
        command: CreateFact,
        *,
        operation_id: str | None = None,
        actor: str = "manual",
    ) -> MemoryFact:
        with self.connection() as connection:
            existing = self._find_active_hash(
                connection, command.scope, content_hash(command.content)
            )
            if existing:
                return self._reinforce(
                    connection,
                    existing,
                    confidence=command.confidence,
                    evidence_message_ids=command.evidence_message_ids,
                    operation_id=operation_id,
                    actor=actor,
                )
            return self._insert_fact(
                connection, command, operation_id=operation_id, actor=actor
            )

    def get(self, fact_id: str) -> MemoryFact | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
        return self._fact_from_row(row) if row else None

    def list_facts(
        self,
        scope: MemoryScope,
        *,
        status: FactStatus | None = FactStatus.ACTIVE,
        limit: int = 100,
    ) -> list[MemoryFact]:
        query = "SELECT * FROM memory_facts WHERE scope_key = ?"
        params: list[Any] = [scope.key]
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self.connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def history(self, fact_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT revision, event_type, snapshot_json, operation_id, actor, created_at
                   FROM memory_history WHERE fact_id = ?
                   ORDER BY history_id DESC LIMIT ?""",
                (fact_id, limit),
            ).fetchall()
        return [
            {
                "revision": row["revision"],
                "event_type": row["event_type"],
                "snapshot": json.loads(row["snapshot_json"]),
                "operation_id": row["operation_id"],
                "actor": row["actor"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _require_fact(
        self, connection: sqlite3.Connection, fact_id: str
    ) -> MemoryFact:
        row = connection.execute(
            "SELECT * FROM memory_facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise FactNotFoundError(fact_id)
        return self._fact_from_row(row)

    @staticmethod
    def _find_active_hash(
        connection: sqlite3.Connection, scope: MemoryScope, digest: str
    ) -> MemoryFact | None:
        row = connection.execute(
            """SELECT * FROM memory_facts
               WHERE scope_key = ? AND content_hash = ? AND status = 'active'""",
            (scope.key, digest),
        ).fetchone()
        return MemoryRepository._fact_from_row(row) if row else None

    def update(
        self,
        fact_id: str,
        command: UpdateFact,
        *,
        expected_revision: int,
        operation_id: str | None = None,
        actor: str = "manual",
    ) -> MemoryFact:
        with self.connection() as connection:
            current = self._require_fact(connection, fact_id)
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    f"fact {fact_id} is revision {current.revision}, expected {expected_revision}"
                )
            updates = command.model_dump(exclude_none=True)
            content = " ".join(str(updates.get("content", current.content)).split()).strip()
            category = str(updates.get("category", current.category)).strip() or "general"
            confidence = float(updates.get("confidence", current.confidence))
            expected_days = updates.get("expected_valid_days", current.expected_valid_days)
            metadata = updates.get("metadata", current.metadata)
            now = utc_now()
            cursor = connection.execute(
                """UPDATE memory_facts SET content = ?, content_hash = ?, category = ?,
                     confidence = ?, expected_valid_days = ?, metadata_json = ?,
                     revision = revision + 1, updated_at = ?
                   WHERE fact_id = ? AND revision = ?""",
                (
                    content,
                    content_hash(content),
                    category,
                    confidence,
                    expected_days,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    fact_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError(fact_id)
            updated = self._require_fact(connection, fact_id)
            self._touch_manifest(connection, updated.scope)
            self._record_history(
                connection, updated, "updated", operation_id=operation_id, actor=actor
            )
            self._queue_index(connection, fact_id)
            return updated

    def _set_status(
        self,
        connection: sqlite3.Connection,
        fact: MemoryFact,
        status: FactStatus,
        *,
        superseded_by: str | None = None,
        operation_id: str | None,
        actor: str,
    ) -> MemoryFact:
        if fact.status == status and fact.superseded_by == superseded_by:
            return fact
        connection.execute(
            """UPDATE memory_facts SET status = ?, superseded_by = ?,
                 revision = revision + 1, updated_at = ? WHERE fact_id = ?""",
            (status.value, superseded_by, utc_now(), fact.fact_id),
        )
        updated = self._require_fact(connection, fact.fact_id)
        self._touch_manifest(connection, updated.scope)
        self._record_history(
            connection,
            updated,
            status.value,
            operation_id=operation_id,
            actor=actor,
        )
        self._queue_index(connection, fact.fact_id)
        return updated

    def delete(
        self,
        fact_id: str,
        *,
        expected_revision: int | None = None,
        operation_id: str | None = None,
        actor: str = "manual",
    ) -> MemoryFact:
        with self.connection() as connection:
            fact = self._require_fact(connection, fact_id)
            if expected_revision is not None and fact.revision != expected_revision:
                raise RevisionConflictError(fact_id)
            return self._set_status(
                connection,
                fact,
                FactStatus.RETRACTED,
                operation_id=operation_id,
                actor=actor,
            )

    def _reinforce(
        self,
        connection: sqlite3.Connection,
        fact: MemoryFact,
        *,
        confidence: float,
        evidence_message_ids: list[str],
        operation_id: str | None,
        actor: str,
    ) -> MemoryFact:
        evidence = list(dict.fromkeys([*fact.evidence_message_ids, *evidence_message_ids]))
        # Repeated independent observations increase confidence conservatively.
        adjusted = 1.0 - (1.0 - fact.confidence) * (1.0 - confidence * 0.5)
        connection.execute(
            """UPDATE memory_facts SET confidence = ?, confirmation_count = confirmation_count + 1,
                 last_confirmed_at = ?, evidence_message_ids_json = ?,
                 revision = revision + 1, updated_at = ? WHERE fact_id = ?""",
            (
                min(1.0, adjusted),
                utc_now(),
                json.dumps(evidence, ensure_ascii=False),
                utc_now(),
                fact.fact_id,
            ),
        )
        updated = self._require_fact(connection, fact.fact_id)
        self._touch_manifest(connection, updated.scope)
        self._record_history(
            connection, updated, "reinforced", operation_id=operation_id, actor=actor
        )
        return updated

    def reinforce(
        self,
        fact_id: str,
        *,
        confidence: float = 0.8,
        evidence_message_ids: list[str] | None = None,
        operation_id: str | None = None,
        actor: str = "manual",
    ) -> MemoryFact:
        with self.connection() as connection:
            fact = self._require_fact(connection, fact_id)
            if fact.status != FactStatus.ACTIVE:
                raise ValueError("only active facts can be reinforced")
            return self._reinforce(
                connection,
                fact,
                confidence=confidence,
                evidence_message_ids=evidence_message_ids or [],
                operation_id=operation_id,
                actor=actor,
            )

    def enqueue_capture(self, capture: CaptureRecord) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_capture_outbox
                   (capture_id, scope_key, user_id, project_id, agent_name, scope_type,
                    thread_id, turn_id, messages_json, signals_json, status, attempts,
                    available_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                (
                    capture.capture_id,
                    capture.scope.key,
                    capture.scope.user_id,
                    capture.scope.project_id,
                    capture.scope.agent_name,
                    capture.scope.scope_type.value,
                    capture.thread_id,
                    capture.turn_id,
                    json.dumps(capture.messages, ensure_ascii=False),
                    json.dumps(capture.signals, ensure_ascii=False),
                    capture.created_at,
                    capture.created_at,
                ),
            )
            return cursor.rowcount == 1

    def pending_captures(self, limit: int = 20) -> list[CaptureRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT current.* FROM memory_capture_outbox current
                   WHERE current.status = 'pending' AND current.available_at <= ?
                     AND NOT EXISTS (
                       SELECT 1 FROM memory_capture_outbox earlier
                       WHERE earlier.scope_key = current.scope_key
                         AND earlier.thread_id = current.thread_id
                         AND earlier.status = 'pending'
                         AND (
                           earlier.created_at < current.created_at OR
                           (earlier.created_at = current.created_at AND earlier.capture_id < current.capture_id)
                         )
                     )
                   ORDER BY current.created_at LIMIT ?""",
                (utc_now(), limit),
            ).fetchall()
        return [
            CaptureRecord(
                capture_id=row["capture_id"],
                scope=MemoryScope(
                    scope_type=ScopeType(row["scope_type"]),
                    user_id=row["user_id"],
                    project_id=row["project_id"],
                    agent_name=row["agent_name"],
                ),
                thread_id=row["thread_id"],
                turn_id=row["turn_id"],
                messages=json.loads(row["messages_json"]),
                signals=json.loads(row["signals_json"]),
                attempts=int(row["attempts"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def mark_capture_failed(self, capture_id: str, error: str) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT attempts FROM memory_capture_outbox WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
            if row is None:
                return
            attempts = int(row["attempts"]) + 1
            delay = min(3600, 2 ** min(attempts, 10))
            available = (datetime.fromisoformat(utc_now()) + timedelta(seconds=delay)).isoformat()
            connection.execute(
                """UPDATE memory_capture_outbox SET attempts = ?, last_error = ?,
                     available_at = ? WHERE capture_id = ?""",
                (attempts, error[:1000], available, capture_id),
            )

    def _complete_capture(
        self, connection: sqlite3.Connection, capture: CaptureRecord
    ) -> None:
        connection.execute(
            """UPDATE memory_capture_outbox SET status = 'completed', completed_at = ?,
                 last_error = NULL WHERE capture_id = ?""",
            (utc_now(), capture.capture_id),
        )
        connection.execute(
            """INSERT INTO memory_watermarks
               (scope_key, thread_id, last_capture_id, last_turn_id, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(scope_key, thread_id) DO UPDATE SET
                 last_capture_id = excluded.last_capture_id,
                 last_turn_id = excluded.last_turn_id,
                 updated_at = excluded.updated_at""",
            (
                capture.scope.key,
                capture.thread_id,
                capture.capture_id,
                capture.turn_id,
                utc_now(),
            ),
        )

    def apply_changeset(
        self,
        capture: CaptureRecord,
        changeset: ChangeSet,
        *,
        minimum_confidence: float,
    ) -> ApplyResult:
        """Apply extractor proposals atomically and advance the watermark last."""

        with self.write_connection() as connection:
            prior = connection.execute(
                "SELECT result_json FROM memory_applied_operations WHERE operation_id = ?",
                (changeset.operation_id,),
            ).fetchone()
            if prior:
                self._complete_capture(connection, capture)
                return ApplyResult.model_validate_json(prior["result_json"])

            result = ApplyResult(operation_id=changeset.operation_id)
            allowed_evidence = {
                item.get("message_id")
                for item in capture.messages
                if item.get("role") == "user" and item.get("message_id")
            }
            for candidate in changeset.candidates:
                result = self._apply_candidate(
                    connection,
                    capture,
                    candidate,
                    result,
                    allowed_evidence=allowed_evidence,
                    minimum_confidence=minimum_confidence,
                    operation_id=changeset.operation_id,
                )
            connection.execute(
                """INSERT INTO memory_applied_operations(operation_id, result_json, applied_at)
                   VALUES (?, ?, ?)""",
                (changeset.operation_id, result.model_dump_json(), utc_now()),
            )
            self._complete_capture(connection, capture)
            return result

    def _apply_candidate(
        self,
        connection: sqlite3.Connection,
        capture: CaptureRecord,
        candidate: FactCandidate,
        result: ApplyResult,
        *,
        allowed_evidence: set[str | None],
        minimum_confidence: float,
        operation_id: str,
    ) -> ApplyResult:
        # Defense in depth: callers may construct a ChangeSet without going
        # through MemoryExtractor.sanitize_candidates(). Never persist an
        # obvious credential from an automatic conversation capture.
        from .extractor import MemoryExtractor

        if MemoryExtractor._looks_secret(candidate.content):
            return result.model_copy(
                update={"skipped": [*result.skipped, "sensitive_content"]}
            )
        if candidate.confidence < minimum_confidence:
            return result.model_copy(update={"skipped": [*result.skipped, "low_confidence"]})
        evidence = [item for item in candidate.evidence_message_ids if item in allowed_evidence]
        if not evidence:
            return result.model_copy(update={"skipped": [*result.skipped, "unsupported"]})
        scope = capture.scope.model_copy(
            update={"scope_type": ScopeType.PROJECT if candidate.scope == "project" else ScopeType.USER}
        )
        target: MemoryFact | None = None
        if candidate.target_id:
            try:
                target = self._require_fact(connection, candidate.target_id)
            except FactNotFoundError:
                target = None
            if target is not None and target.scope.key != scope.key:
                return result.model_copy(update={"skipped": [*result.skipped, "scope_mismatch"]})
        duplicate = self._find_active_hash(connection, scope, content_hash(candidate.content))

        # Explicit destructive relations must win over hash deduplication.  A
        # retract candidate intentionally repeats the old fact's content, so a
        # blanket ``or duplicate`` here would turn withdrawal into reinforcement.
        if candidate.relation == "retract":
            if target is None or target.status != FactStatus.ACTIVE:
                return result.model_copy(update={"skipped": [*result.skipped, "missing_target"]})
            updated = self._set_status(
                connection,
                target,
                FactStatus.RETRACTED,
                operation_id=operation_id,
                actor="extractor",
            )
            return result.model_copy(update={"retracted": [*result.retracted, updated.fact_id]})

        if candidate.relation in {"duplicate", "reinforce"} or (
            candidate.relation == "new" and duplicate is not None
        ):
            target = target or duplicate
            if target is None or target.status != FactStatus.ACTIVE:
                return result.model_copy(update={"skipped": [*result.skipped, "missing_target"]})
            updated = self._reinforce(
                connection,
                target,
                confidence=candidate.confidence,
                evidence_message_ids=evidence,
                operation_id=operation_id,
                actor="extractor",
            )
            return result.model_copy(update={"reinforced": [*result.reinforced, updated.fact_id]})

        # A correction may converge on a replacement fact that is already
        # active.  Reuse it and close the old fact instead of violating the
        # active-hash uniqueness constraint or silently treating the correction
        # as a plain reinforcement.
        if candidate.relation == "supersede" and duplicate is not None:
            if target is None or target.status != FactStatus.ACTIVE:
                return result.model_copy(
                    update={"skipped": [*result.skipped, "missing_supersede_target"]}
                )
            if duplicate.fact_id == target.fact_id:
                return result.model_copy(
                    update={"skipped": [*result.skipped, "supersede_same_content"]}
                )
            replacement = self._reinforce(
                connection,
                duplicate,
                confidence=candidate.confidence,
                evidence_message_ids=evidence,
                operation_id=operation_id,
                actor="extractor",
            )
            old = self._set_status(
                connection,
                target,
                FactStatus.SUPERSEDED,
                superseded_by=replacement.fact_id,
                operation_id=operation_id,
                actor="extractor",
            )
            return result.model_copy(
                update={
                    "reinforced": [*result.reinforced, replacement.fact_id],
                    "superseded": [*result.superseded, old.fact_id],
                }
            )

        command = CreateFact(
            scope=scope,
            content=candidate.content,
            category=candidate.category,
            confidence=candidate.confidence,
            source_type=SourceType.CONVERSATION,
            source_thread_id=capture.thread_id,
            evidence_message_ids=evidence,
            expected_valid_days=candidate.expected_valid_days,
            metadata={**candidate.metadata, "capture_id": capture.capture_id},
        )
        created = self._insert_fact(
            connection, command, operation_id=operation_id, actor="extractor"
        )
        update = {"created": [*result.created, created.fact_id]}
        if candidate.relation == "supersede":
            if target is None or target.status != FactStatus.ACTIVE:
                # Preserve the supported new fact, but record that its relation was invalid.
                update["skipped"] = [*result.skipped, "missing_supersede_target"]
            else:
                old = self._set_status(
                    connection,
                    target,
                    FactStatus.SUPERSEDED,
                    superseded_by=created.fact_id,
                    operation_id=operation_id,
                    actor="extractor",
                )
                update["superseded"] = [*result.superseded, old.fact_id]
        return result.model_copy(update=update)

    def queue_embedding_migration(self, model: str) -> int:
        """Queue active rows whose persisted vector belongs to an older model."""

        with self.connection() as connection:
            now = utc_now()
            cursor = connection.execute(
                """INSERT INTO memory_index_outbox(fact_id, attempts, available_at, created_at)
                   SELECT f.fact_id, 0, ?, ? FROM memory_facts f
                   LEFT JOIN memory_embeddings e ON e.fact_id = f.fact_id
                   WHERE f.status = 'active' AND (e.fact_id IS NULL OR e.model != ?)
                   ON CONFLICT(fact_id) DO NOTHING""",
                (now, now, model),
            )
            connection.execute(
                """INSERT INTO memory_document_index_outbox
                   (document_id, attempts, available_at, created_at)
                   SELECT d.document_id, 0, ?, ? FROM memory_documents d
                   LEFT JOIN memory_document_embeddings e ON e.document_id = d.document_id
                   WHERE e.document_id IS NULL OR e.model != ?
                   ON CONFLICT(document_id) DO NOTHING""",
                (now, now, model),
            )
            return max(0, int(cursor.rowcount))

    def process_index_outbox(self, provider: Any, limit: int = 100) -> tuple[int, int]:
        """Materialize FTS and provider vectors without holding a DB write lock."""

        with self.connection() as connection:
            rows = connection.execute(
                """SELECT o.fact_id, o.attempts, f.content, f.content_hash,
                          f.category, f.status
                   FROM memory_index_outbox o
                   JOIN memory_facts f ON f.fact_id = o.fact_id
                   WHERE o.available_at <= ? ORDER BY o.created_at DESC LIMIT ?""",
                (utc_now(), limit),
            ).fetchall()
        indexed = 0
        failed = 0
        active = [row for row in rows if row["status"] == FactStatus.ACTIVE.value]
        vectors: dict[str, list[float]] = {}
        if active:
            try:
                encoded = provider.encode_documents([row["content"] for row in active])
                vectors = {
                    row["fact_id"]: vector
                    for row, vector in zip(active, encoded, strict=True)
                }
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"[:1000]
                with self.connection() as connection:
                    for row in active:
                        connection.execute(
                            """UPDATE memory_index_outbox SET attempts = attempts + 1,
                                 last_error = ?, available_at = ? WHERE fact_id = ?""",
                            (message, self._retry_at(int(row["attempts"])), row["fact_id"]),
                        )
                return 0, len(active)
        for row in rows:
            try:
                with self.connection() as connection:
                    connection.execute("DELETE FROM memory_fts WHERE fact_id = ?", (row["fact_id"],))
                    if row["status"] == FactStatus.ACTIVE.value:
                        connection.execute(
                            """INSERT INTO memory_fts
                               (fact_id, content, category, search_text)
                               VALUES (?, ?, ?, ?)""",
                            (
                                row["fact_id"],
                                row["content"],
                                row["category"],
                                lexical_index_text(row["content"], row["category"]),
                            ),
                        )
                        vector = vectors[row["fact_id"]]
                        connection.execute(
                            """INSERT INTO memory_embeddings
                               (fact_id, model, dimensions, vector, content_hash, indexed_at)
                               VALUES (?, ?, ?, ?, ?, ?)
                               ON CONFLICT(fact_id) DO UPDATE SET
                                 model = excluded.model, dimensions = excluded.dimensions,
                                 vector = excluded.vector, content_hash = excluded.content_hash,
                                 indexed_at = excluded.indexed_at""",
                            (
                                row["fact_id"],
                                provider.model_id,
                                len(vector),
                                pack_vector(vector),
                                row["content_hash"],
                                utc_now(),
                            ),
                        )
                    else:
                        connection.execute(
                            "DELETE FROM memory_embeddings WHERE fact_id = ?", (row["fact_id"],)
                        )
                    # Do not acknowledge a stale projection if the fact changed
                    # while a model batch was being computed.
                    current = connection.execute(
                        "SELECT content_hash, status FROM memory_facts WHERE fact_id = ?",
                        (row["fact_id"],),
                    ).fetchone()
                    if current and current["content_hash"] == row["content_hash"]:
                        connection.execute(
                            "DELETE FROM memory_index_outbox WHERE fact_id = ?", (row["fact_id"],)
                        )
                    indexed += 1
            except Exception as exc:  # one bad row must not block the queue
                with self.connection() as connection:
                    connection.execute(
                        """UPDATE memory_index_outbox SET attempts = attempts + 1,
                             last_error = ?, available_at = ? WHERE fact_id = ?""",
                        (
                            f"{type(exc).__name__}: {exc}"[:1000],
                            self._retry_at(int(row["attempts"])),
                            row["fact_id"],
                        ),
                    )
                failed += 1
        return indexed, failed

    def upsert_document(self, document: MemoryDocument, *, immutable: bool = False) -> bool:
        normalized = " ".join(document.content.split()).strip()
        digest = content_hash(normalized)
        now = utc_now()
        with self.connection() as connection:
            existing = connection.execute(
                """SELECT content_hash, document_type, scope_key, metadata_json
                   FROM memory_documents WHERE document_id = ?""",
                (document.document_id,),
            ).fetchone()
            if existing and immutable:
                if existing["content_hash"] != digest:
                    raise ValueError(
                        f"immutable memory document changed: {document.document_id}"
                    )
                return False
            serialized_metadata = json.dumps(document.metadata, ensure_ascii=False)
            if (
                existing
                and existing["content_hash"] == digest
                and existing["document_type"] == document.document_type
                and existing["scope_key"] == document.scope.key
                and existing["metadata_json"] == serialized_metadata
            ):
                return False
            connection.execute(
                """INSERT INTO memory_documents
                   (document_id, document_type, scope_key, scope_type, user_id,
                    project_id, agent_name, content, content_hash, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                     document_type = excluded.document_type,
                     scope_key = excluded.scope_key, scope_type = excluded.scope_type,
                     user_id = excluded.user_id, project_id = excluded.project_id,
                     agent_name = excluded.agent_name, content = excluded.content,
                     content_hash = excluded.content_hash,
                     metadata_json = excluded.metadata_json,
                     updated_at = excluded.updated_at""",
                (
                    document.document_id,
                    document.document_type,
                    document.scope.key,
                    document.scope.scope_type.value,
                    document.scope.user_id,
                    document.scope.project_id,
                    document.scope.agent_name,
                    normalized,
                    digest,
                    serialized_metadata,
                    document.created_at,
                    now,
                ),
            )
            connection.execute(
                """INSERT INTO memory_document_index_outbox
                   (document_id, attempts, available_at, created_at)
                   VALUES (?, 0, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET attempts = 0,
                     last_error = NULL, available_at = excluded.available_at,
                     created_at = excluded.created_at""",
                (document.document_id, now, now),
            )
        return True

    def process_document_index_outbox(
        self, provider: Any, limit: int = 100
    ) -> tuple[int, int]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT o.document_id, o.attempts, d.content, d.content_hash,
                          d.document_type
                   FROM memory_document_index_outbox o
                   JOIN memory_documents d ON d.document_id = o.document_id
                   WHERE o.available_at <= ? ORDER BY o.created_at DESC LIMIT ?""",
                (utc_now(), limit),
            ).fetchall()
        if not rows:
            return 0, 0
        try:
            encoded = provider.encode_documents([row["content"] for row in rows])
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:1000]
            with self.connection() as connection:
                for row in rows:
                    connection.execute(
                        """UPDATE memory_document_index_outbox
                           SET attempts = attempts + 1, last_error = ?, available_at = ?
                           WHERE document_id = ?""",
                        (
                            message,
                            self._retry_at(int(row["attempts"])),
                            row["document_id"],
                        ),
                    )
            return 0, len(rows)
        indexed = failed = 0
        for row, vector in zip(rows, encoded, strict=True):
            try:
                with self.connection() as connection:
                    connection.execute(
                        "DELETE FROM memory_document_fts WHERE document_id = ?",
                        (row["document_id"],),
                    )
                    connection.execute(
                        """INSERT INTO memory_document_fts
                           (document_id, content, document_type, search_text)
                           VALUES (?, ?, ?, ?)""",
                        (
                            row["document_id"],
                            row["content"],
                            row["document_type"],
                            lexical_index_text(row["content"], row["document_type"]),
                        ),
                    )
                    connection.execute(
                        """INSERT INTO memory_document_embeddings
                           (document_id, model, dimensions, vector, content_hash, indexed_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(document_id) DO UPDATE SET
                             model = excluded.model, dimensions = excluded.dimensions,
                             vector = excluded.vector, content_hash = excluded.content_hash,
                             indexed_at = excluded.indexed_at""",
                        (
                            row["document_id"],
                            provider.model_id,
                            len(vector),
                            pack_vector(vector),
                            row["content_hash"],
                            utc_now(),
                        ),
                    )
                    current = connection.execute(
                        "SELECT content_hash FROM memory_documents WHERE document_id = ?",
                        (row["document_id"],),
                    ).fetchone()
                    if current and current["content_hash"] == row["content_hash"]:
                        connection.execute(
                            "DELETE FROM memory_document_index_outbox WHERE document_id = ?",
                            (row["document_id"],),
                        )
                indexed += 1
            except Exception as exc:
                with self.connection() as connection:
                    connection.execute(
                        """UPDATE memory_document_index_outbox
                           SET attempts = attempts + 1, last_error = ?, available_at = ?
                           WHERE document_id = ?""",
                        (
                            f"{type(exc).__name__}: {exc}"[:1000],
                            self._retry_at(int(row["attempts"])),
                            row["document_id"],
                        ),
                    )
                failed += 1
        return indexed, failed

    def fact_vector_rows(
        self, scope: MemoryScope, *, model: str, dimensions: int
    ) -> list[tuple[str, bytes, int]]:
        scope_keys = self.authorized_scope_keys(scope)
        placeholders = ", ".join("?" for _ in scope_keys)
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT f.fact_id AS identifier, e.vector, e.dimensions
                    FROM memory_facts f JOIN memory_embeddings e ON e.fact_id = f.fact_id
                    WHERE f.scope_key IN ({placeholders}) AND f.status = 'active'
                      AND e.model = ? AND e.dimensions = ?""",
                (*scope_keys, model, dimensions),
            ).fetchall()
        return [(row["identifier"], row["vector"], row["dimensions"]) for row in rows]

    def document_vector_rows(
        self,
        scope: MemoryScope,
        *,
        model: str,
        dimensions: int,
        document_types: set[str] | None = None,
    ) -> list[tuple[str, bytes, int]]:
        scope_keys = self.authorized_scope_keys(scope)
        clauses = [f"d.scope_key IN ({', '.join('?' for _ in scope_keys)})"]
        params: list[Any] = [*scope_keys]
        if document_types:
            clauses.append(
                f"d.document_type IN ({', '.join('?' for _ in document_types)})"
            )
            params.extend(sorted(document_types))
        clauses.extend(["e.model = ?", "e.dimensions = ?"])
        params.extend([model, dimensions])
        with self.connection() as connection:
            rows = connection.execute(
                f"""SELECT d.document_id AS identifier, e.vector, e.dimensions
                    FROM memory_documents d JOIN memory_document_embeddings e
                      ON e.document_id = d.document_id
                    WHERE {' AND '.join(clauses)}""",
                params,
            ).fetchall()
        return [(row["identifier"], row["vector"], row["dimensions"]) for row in rows]

    def retrieval_rows(
        self,
        scope: MemoryScope,
        query: str,
        *,
        limit: int = 100,
        candidate_ids: list[str] | None = None,
        lexical_limit: int = 50,
    ) -> list[tuple[MemoryFact, bytes | None, int | None, float]]:
        """Return authorized active candidates; scoring stays in the service."""

        scope_keys = self.authorized_scope_keys(scope)
        placeholders = ", ".join("?" for _ in scope_keys)
        with self.connection() as connection:
            fts_scores: dict[str, float] = {}
            expression = fts_query_expression(query)
            if expression and self.fts_enabled:
                try:
                    matches = connection.execute(
                        f"""SELECT search.fact_id, bm25(memory_fts) AS rank
                            FROM memory_fts search
                            JOIN memory_facts fact ON fact.fact_id = search.fact_id
                            WHERE memory_fts MATCH ?
                              AND fact.scope_key IN ({placeholders})
                              AND fact.status = 'active'
                            ORDER BY rank LIMIT ?""",
                        (expression, *scope_keys, lexical_limit),
                    ).fetchall()
                    fts_scores = {
                        row["fact_id"]: 1.0 / (rank + 1)
                        for rank, row in enumerate(matches)
                    }
                except sqlite3.OperationalError:
                    # Malformed tokenizer input must degrade to deterministic
                    # overlap scoring rather than fail retrieval.
                    fts_scores = {}
            selected_ids = list(dict.fromkeys([*fts_scores, *(candidate_ids or [])]))
            if selected_ids:
                id_placeholders = ", ".join("?" for _ in selected_ids)
                rows = connection.execute(
                    f"""SELECT f.*, e.vector AS embedding_vector,
                               e.dimensions AS embedding_dimensions
                       FROM memory_facts f LEFT JOIN memory_embeddings e ON e.fact_id = f.fact_id
                       WHERE f.scope_key IN ({placeholders}) AND f.status = 'active'
                         AND f.fact_id IN ({id_placeholders})
                       ORDER BY f.updated_at DESC LIMIT ?""",
                    (*scope_keys, *selected_ids, limit),
                ).fetchall()
            else:
                # Recent rows preserve lexical-overlap behavior when FTS5 is
                # unavailable and provide a safe fallback during reindexing.
                rows = connection.execute(
                    f"""SELECT f.*, e.vector AS embedding_vector,
                               e.dimensions AS embedding_dimensions
                       FROM memory_facts f LEFT JOIN memory_embeddings e ON e.fact_id = f.fact_id
                       WHERE f.scope_key IN ({placeholders}) AND f.status = 'active'
                       ORDER BY f.updated_at DESC LIMIT ?""",
                    (*scope_keys, limit),
                ).fetchall()
        return [
            (
                self._fact_from_row(row),
                row["embedding_vector"],
                row["embedding_dimensions"],
                fts_scores.get(row["fact_id"], 0.0),
            )
            for row in rows
        ]

    def document_candidates(
        self,
        scope: MemoryScope,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        document_types: set[str] | None = None,
        lexical_limit: int = 50,
        limit: int = 100,
    ) -> list[tuple[MemoryDocument, float]]:
        scope_keys = self.authorized_scope_keys(scope)
        placeholders = ", ".join("?" for _ in scope_keys)
        fts_scores: dict[str, float] = {}
        with self.connection() as connection:
            expression = fts_query_expression(query)
            if expression and self.document_fts_enabled:
                type_clause = ""
                type_params: list[str] = []
                if document_types:
                    type_clause = f" AND d.document_type IN ({', '.join('?' for _ in document_types)})"
                    type_params = sorted(document_types)
                try:
                    matches = connection.execute(
                        f"""SELECT search.document_id, bm25(memory_document_fts) AS rank
                            FROM memory_document_fts search
                            JOIN memory_documents d ON d.document_id = search.document_id
                            WHERE memory_document_fts MATCH ?
                              AND d.scope_key IN ({placeholders}) {type_clause}
                            ORDER BY rank LIMIT ?""",
                        (expression, *scope_keys, *type_params, lexical_limit),
                    ).fetchall()
                    fts_scores = {
                        row["document_id"]: 1.0 / (rank + 1)
                        for rank, row in enumerate(matches)
                    }
                except sqlite3.OperationalError:
                    fts_scores = {}
            selected = list(dict.fromkeys([*fts_scores, *(candidate_ids or [])]))
            clauses = [f"scope_key IN ({placeholders})"]
            params: list[Any] = [*scope_keys]
            if document_types:
                clauses.append(
                    f"document_type IN ({', '.join('?' for _ in document_types)})"
                )
                params.extend(sorted(document_types))
            if selected:
                clauses.append(f"document_id IN ({', '.join('?' for _ in selected)})")
                params.extend(selected)
            rows = connection.execute(
                f"""SELECT * FROM memory_documents WHERE {' AND '.join(clauses)}
                    ORDER BY updated_at DESC LIMIT ?""",
                (*params, limit),
            ).fetchall()
        return [
            (self._document_from_row(row), fts_scores.get(row["document_id"], 0.0))
            for row in rows
        ]

    def record_usage(self, fact_ids: list[str]) -> None:
        if not fact_ids:
            return
        with self.connection() as connection:
            for fact_id in fact_ids:
                connection.execute(
                    """INSERT INTO memory_fact_usage(fact_id, retrieval_count, last_retrieved_at)
                       VALUES (?, 1, ?)
                       ON CONFLICT(fact_id) DO UPDATE SET
                         retrieval_count = retrieval_count + 1,
                         last_retrieved_at = excluded.last_retrieved_at""",
                    (fact_id, utc_now()),
                )

    def expire_stale(self, *, default_valid_days: int = 365) -> int:
        now = datetime.fromisoformat(utc_now())
        changed = 0
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM memory_facts
                   WHERE status = 'active' AND source_type != 'manual'"""
            ).fetchall()
            for row in rows:
                fact = self._fact_from_row(row)
                days = fact.expected_valid_days or default_valid_days
                try:
                    confirmed = datetime.fromisoformat(fact.last_confirmed_at)
                except ValueError:
                    continue
                if now - confirmed <= timedelta(days=days):
                    continue
                self._set_status(
                    connection,
                    fact,
                    FactStatus.EXPIRED,
                    operation_id="lifecycle-expire",
                    actor="lifecycle",
                )
                changed += 1
        return changed

    def enforce_capacity(self, scope: MemoryScope, maximum: int) -> int:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM memory_facts WHERE scope_key = ? AND status = 'active'
                   ORDER BY confidence DESC, confirmation_count DESC, updated_at DESC""",
                (scope.key,),
            ).fetchall()
            overflow = rows[maximum:]
            for row in overflow:
                self._set_status(
                    connection,
                    self._fact_from_row(row),
                    FactStatus.EXPIRED,
                    operation_id="lifecycle-capacity",
                    actor="lifecycle",
                )
            return len(overflow)

    def stats(self, scope: MemoryScope | None = None) -> MemoryStats:
        where = " WHERE scope_key = ?" if scope else ""
        params = (scope.key,) if scope else ()
        with self.connection() as connection:
            active = connection.execute(
                f"SELECT COUNT(*) FROM memory_facts{where}" + (" AND status = 'active'" if scope else " WHERE status = 'active'"),
                params,
            ).fetchone()[0]
            inactive = connection.execute(
                f"SELECT COUNT(*) FROM memory_facts{where}" + (" AND status != 'active'" if scope else " WHERE status != 'active'"),
                params,
            ).fetchone()[0]
            captures = connection.execute(
                "SELECT COUNT(*) FROM memory_capture_outbox WHERE status = 'pending'"
            ).fetchone()[0]
            indexes = connection.execute(
                "SELECT COUNT(*) FROM memory_index_outbox"
            ).fetchone()[0]
            history = connection.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0]
            documents = connection.execute(
                """SELECT COUNT(*) FROM memory_document_embeddings e
                   JOIN memory_documents d ON d.document_id = e.document_id"""
                + (" WHERE d.scope_key = ?" if scope else ""),
                params,
            ).fetchone()[0]
            document_indexes = connection.execute(
                "SELECT COUNT(*) FROM memory_document_index_outbox"
            ).fetchone()[0]
        return MemoryStats(
            active_facts=active,
            inactive_facts=inactive,
            pending_captures=captures,
            pending_indexes=indexes,
            history_events=history,
            indexed_documents=documents,
            pending_document_indexes=document_indexes,
        )
