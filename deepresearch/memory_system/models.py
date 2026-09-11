"""Domain contracts for the self-hosted long-term memory system.

These models deliberately describe memory operations instead of exposing the
SQLite layout.  Callers can therefore evolve storage and retrieval without
changing the orchestration layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import utc_now


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeType(StrEnum):
    USER = "user"
    PROJECT = "project"
    AGENT = "agent"


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"


class SourceType(StrEnum):
    MANUAL = "manual"
    CONVERSATION = "conversation"
    RESEARCH_EPISODE = "research_episode"
    CONSOLIDATION = "consolidation"


class MemoryScope(MemoryModel):
    """A typed namespace.  Scope fields are part of every authorization check."""

    scope_type: ScopeType = ScopeType.USER
    user_id: str
    project_id: str
    agent_name: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "MemoryScope":
        if self.scope_type == ScopeType.AGENT and not self.agent_name:
            raise ValueError("agent scope requires agent_name")
        return self

    @property
    def key(self) -> str:
        agent = self.agent_name or "-"
        return f"{self.scope_type.value}:{self.user_id}:{self.project_id}:{agent}"


class PreparedMemoryQuery(MemoryModel):
    """One immutable query embedding shared by related retrieval channels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    vector: list[float]
    embedding_model: str
    dimensions: int = Field(ge=1)
    degraded: bool = False


class MemoryFact(MemoryModel):
    fact_id: str
    scope: MemoryScope
    content: str
    content_hash: str
    category: str = "general"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: FactStatus = FactStatus.ACTIVE
    source_type: SourceType = SourceType.MANUAL
    source_episode_id: str | None = None
    source_thread_id: str | None = None
    evidence_message_ids: list[str] = Field(default_factory=list)
    expected_valid_days: int | None = Field(default=None, ge=1)
    last_confirmed_at: str = Field(default_factory=utc_now)
    confirmation_count: int = Field(default=1, ge=0)
    superseded_by: str | None = None
    consolidated_from: list[str] = Field(default_factory=list)
    revision: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class CreateFact(MemoryModel):
    scope: MemoryScope
    content: str = Field(min_length=1, max_length=4000)
    category: str = "general"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_type: SourceType = SourceType.MANUAL
    source_episode_id: str | None = None
    source_thread_id: str | None = None
    evidence_message_ids: list[str] = Field(default_factory=list)
    expected_valid_days: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateFact(MemoryModel):
    content: str | None = Field(default=None, min_length=1, max_length=4000)
    category: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_valid_days: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] | None = None


class FactCandidate(MemoryModel):
    """One LLM-proposed durable fact before deterministic admission checks."""

    content: str = Field(min_length=1, max_length=4000)
    category: str = "general"
    scope: Literal["user", "project"] = "user"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    relation: Literal["new", "duplicate", "reinforce", "supersede", "retract"] = "new"
    target_id: str | None = None
    evidence_message_ids: list[str] = Field(default_factory=list)
    expected_valid_days: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChangeSet(MemoryModel):
    operation_id: str
    candidates: list[FactCandidate] = Field(default_factory=list)
    extractor_model: str = ""
    created_at: str = Field(default_factory=utc_now)


class SearchScore(MemoryModel):
    lexical: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recency: float = Field(default=0.0, ge=0.0, le=1.0)
    reinforcement: float = Field(default=0.0, ge=0.0, le=1.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)


class MemorySearchResult(MemoryModel):
    fact: MemoryFact
    score: SearchScore
    matched_by: list[str] = Field(default_factory=list)
    explanation: str = ""


class MemoryDocument(MemoryModel):
    """Search projection for episodic and immutable conversation memories."""

    document_id: str
    document_type: Literal["research_episode", "conversation_segment"]
    scope: MemoryScope
    content: str
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class MemoryDocumentSearchResult(MemoryModel):
    document: MemoryDocument
    lexical: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic: float = Field(default=0.0, ge=0.0, le=1.0)
    recency: float = Field(default=0.0, ge=0.0, le=1.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_by: list[str] = Field(default_factory=list)


class CaptureRecord(MemoryModel):
    capture_id: str
    scope: MemoryScope
    thread_id: str
    turn_id: str
    messages: list[dict[str, str]]
    signals: list[str] = Field(default_factory=list)
    attempts: int = 0
    created_at: str = Field(default_factory=utc_now)


class ApplyResult(MemoryModel):
    operation_id: str
    created: list[str] = Field(default_factory=list)
    reinforced: list[str] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)
    retracted: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class MemoryStats(MemoryModel):
    active_facts: int = 0
    inactive_facts: int = 0
    pending_captures: int = 0
    pending_indexes: int = 0
    history_events: int = 0
    indexed_documents: int = 0
    pending_document_indexes: int = 0
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_degraded: bool = False
    embedding_error: str | None = None
