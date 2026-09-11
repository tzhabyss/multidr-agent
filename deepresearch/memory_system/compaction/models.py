"""Typed contracts for durable compaction and runtime memory projection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CompactionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


MustKeepKind = Literal[
    "correction",
    "constraint",
    "decision",
    "numeric",
    "citation",
    "unresolved",
]


class MustKeepSpan(CompactionModel):
    """An exact, validated span from a user message."""

    kind: MustKeepKind
    text: str = Field(min_length=1, max_length=2000)
    message_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class SegmentStateDelta(CompactionModel):
    """Mergeable changes derived from one immutable conversation segment."""

    add_constraints: list[str] = Field(default_factory=list)
    add_decisions: list[str] = Field(default_factory=list)
    add_corrections: list[str] = Field(default_factory=list)
    add_unresolved: list[str] = Field(default_factory=list)
    resolve_unresolved: list[str] = Field(default_factory=list)
    resolve_all_unresolved: bool = False
    retract_constraints: list[str] = Field(default_factory=list)
    retract_decisions: list[str] = Field(default_factory=list)
    supersede_constraints: list[dict[str, str]] = Field(default_factory=list)
    supersede_decisions: list[dict[str, str]] = Field(default_factory=list)


class CompactedSegment(CompactionModel):
    """Multi-resolution representation of one archived conversation slice."""

    schema_version: int = 2
    segment_id: str
    abstract: str = Field(min_length=1, max_length=1200)
    summary: str = Field(min_length=1, max_length=12000)
    active_entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    must_keep_spans: list[MustKeepSpan] = Field(default_factory=list)
    rejected_sensitive_spans: int = Field(default=0, ge=0)
    redacted_sensitive_fields: int = Field(default=0, ge=0)
    state_delta: SegmentStateDelta = Field(default_factory=SegmentStateDelta)
    detail_ref: dict[str, Any] = Field(default_factory=dict)
    compression: dict[str, Any] = Field(default_factory=dict)


class ProjectionResult(CompactionModel):
    """Ephemeral, query- and role-specific view over durable memories."""

    items: list[dict[str, Any]] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)
    raw_estimated_tokens: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    level: Literal[
        "normal", "light", "query_aware", "summarize", "emergency"
    ] = "normal"
