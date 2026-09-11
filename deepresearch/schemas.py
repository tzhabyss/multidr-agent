"""Strong data contracts shared by the graph, runtime, and evaluation code."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .time_utils import now_iso


def utc_now() -> str:
    """Backward-compatible name for the project-local timestamp factory."""
    return now_iso()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FIXED_PROJECT_ID = "deepresearch-default"
DEFAULT_USER_ID = "local-user"


class ConversationMessage(StrictModel):
    message_id: str
    turn_id: str
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: str = Field(default_factory=utc_now)
    run_id: str | None = None


class RollingMemoryState(StrictModel):
    """Deterministic state rebuilt from immutable conversation segments.

    The free-form ``rolling_summary`` remains as a backwards-compatible prompt
    projection.  This model is the authoritative, mergeable representation so
    repeated compaction does not have to summarize an earlier summary.
    """

    revision: int = Field(default=0, ge=0)
    segment_ids: list[str] = Field(default_factory=list)
    segment_abstracts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class SessionMemory(StrictModel):
    user_id: str = DEFAULT_USER_ID
    project_id: str = FIXED_PROJECT_ID
    session_id: str
    title: str | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    rolling_summary: str = ""
    rolling_state: RollingMemoryState = Field(default_factory=RollingMemoryState)
    active_entities: list[str] = Field(default_factory=list)
    current_intent: str = ""
    user_corrections: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    pending_clarification: str | None = None
    related_run_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class ResearchBrief(StrictModel):
    question: str
    objective: str
    assumptions: list[str] = Field(default_factory=list)
    resolved_references: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)


class ClarifierDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: Literal["clarify", "research", "respond_from_context"] = "research"
    research_brief: ResearchBrief | None = None
    question: str | None = None
    response: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    user_corrections: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    memory_refs: list[str] = Field(default_factory=list)


class MemoryEpisode(StrictModel):
    episode_id: str
    user_id: str
    project_id: str
    session_id: str
    run_id: str
    research_brief: str
    status: str
    summary: str
    verified_claim_ids: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    successful_queries: list[str] = Field(default_factory=list)
    failed_queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    user_corrections: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None
    created_at: str = Field(default_factory=utc_now)


class MemoryPacket(StrictModel):
    role: Literal["clarifier", "planner", "worker", "synthesizer"]
    query: str = ""
    session_summary: str = ""
    recent_dialogue: list[dict[str, str]] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    current_state: dict[str, Any] = Field(default_factory=dict)
    relevant_episodes: list[dict[str, Any]] = Field(default_factory=list)
    historical_segments: list[dict[str, Any]] = Field(default_factory=list)
    successful_queries: list[str] = Field(default_factory=list)
    failed_queries: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    relevant_claims: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)
    projection_level: Literal[
        "normal", "light", "query_aware", "summarize", "emergency"
    ] = "normal"
    budget_tokens: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)


class ResearchMode(str, Enum):
    """Caller-selected research depth; legacy values are normalized safely."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    MAX = "max"
    # Backwards-compatible names used by old callers and checkpoints.
    RESEARCH = "standard"
    DEEP_SEARCH = "standard"
    DEEP_REPORT = "standard"

    @classmethod
    def _missing_(cls, value: object):
        normalized = str(value).strip().lower()
        if normalized == "research":
            return cls.STANDARD
        if normalized in {"deep_search", "deep_report"}:
            return cls.STANDARD
        return None


class QueryType(str, Enum):
    STRAIGHTFORWARD = "straightforward"
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED_VERIFIED = "completed_verified"
    COMPLETED_PARTIAL = "completed_partial"
    FAILED_NO_EVIDENCE = "failed_no_evidence"
    FAILED_BUDGET_EXHAUSTED = "failed_budget_exhausted"
    FAILED_POLICY = "failed_policy"
    FAILED_SYSTEM = "failed_system"
    CANCELLED = "cancelled"


class TaskSufficiency(StrictModel):
    """Resolved minimum evidence floor for one research task."""

    min_queries: int = Field(default=1, ge=1)
    min_query_coverage: int = Field(default=1, ge=1)
    min_successful_extractions: int = Field(default=1, ge=1)
    min_effective_sources: int = Field(default=1, ge=1)
    min_independent_domains: int = Field(default=1, ge=1)
    min_evidence: int = Field(default=1, ge=1)
    max_source_quality_tier: int = Field(default=3, ge=1, le=4)


class TaskSufficiencyOverrides(StrictModel):
    """Optional request-level overrides merged onto mode defaults."""

    min_queries: int | None = Field(default=None, ge=1)
    min_query_coverage: int | None = Field(default=None, ge=1)
    min_successful_extractions: int | None = Field(default=None, ge=1)
    min_effective_sources: int | None = Field(default=None, ge=1)
    min_independent_domains: int | None = Field(default=None, ge=1)
    min_evidence: int | None = Field(default=None, ge=1)
    max_source_quality_tier: int | None = Field(default=None, ge=1, le=4)


class ResearchOptions(StrictModel):
    task_sufficiency: TaskSufficiencyOverrides = Field(
        default_factory=TaskSufficiencyOverrides
    )
    max_rounds: int = Field(default=3, ge=1, le=6)
    max_worker_iterations: int = Field(default=4, ge=1, le=12)
    max_workers: int = Field(default=6, ge=1, le=16)


class ResearchConstraints(StrictModel):
    locale: str = "zh-CN"
    date_boundary: str | None = None
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    blocked_urls: list[str] = Field(default_factory=list)
    required_dimensions: list[str] = Field(default_factory=list)
    excluded_dimensions: list[str] = Field(default_factory=list)


class ResearchRequest(StrictModel):
    question: str
    mode: ResearchMode = ResearchMode.RESEARCH
    profile_id: str = "general_research"
    source_urls: list[str] = Field(default_factory=list)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    options: ResearchOptions = Field(default_factory=ResearchOptions)
    run_id: str | None = None
    user_id: str = DEFAULT_USER_ID
    project_id: str = FIXED_PROJECT_ID
    session_id: str | None = None
    turn_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value

    @field_validator("run_id", "session_id", "turn_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("run_id may contain only letters, digits, '-' and '_'")
        return value

    @field_validator("project_id")
    @classmethod
    def validate_fixed_project(cls, value: str) -> str:
        if value != FIXED_PROJECT_ID:
            raise ValueError(f"project_id is fixed to {FIXED_PROJECT_ID!r} in v1")
        return value


class ResearchScope(StrictModel):
    question: str
    mode: ResearchMode
    profile_id: str
    objective: str
    required_dimensions: list[str]
    suggested_dimensions: list[str] = Field(default_factory=list)
    current_date: str
    date_boundary: str | None = None
    locale: str = "zh-CN"
    assumptions: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    options: ResearchOptions = Field(default_factory=ResearchOptions)


class TaskBudget(StrictModel):
    max_tokens: int | None = Field(default=None, ge=0, exclude=True)
    max_searches: int = Field(ge=0)
    max_fetches: int = Field(ge=0)
    max_wall_time_seconds: int = Field(gt=0)


class RunBudget(StrictModel):
    max_total_tokens: int | None = Field(default=None, gt=0, exclude=True)
    max_total_searches: int = Field(gt=0)
    max_total_fetches: int = Field(gt=0)
    max_wall_time_seconds: int = Field(gt=0)
    max_cost_usd: float | None = None
    planner_reserved_tokens: int = Field(default=0, ge=0, exclude=True)
    synthesizer_reserved_tokens: int = Field(default=0, ge=0, exclude=True)

    @classmethod
    def default(cls) -> "RunBudget":
        return cls(
            max_total_searches=200,
            max_total_fetches=336,
            max_wall_time_seconds=1500,
            max_cost_usd=None,
        )

    @classmethod
    def for_mode(cls, mode: ResearchMode | None = None) -> "RunBudget":
        """Backward-compatible constructor with no mode-dependent behavior."""
        return cls.default()


class PlannerCandidateCase(StrictModel):
    """Unverified Planner lead for one repeated row or case.

    These values guide retrieval and coverage accounting. They are not claims
    and cannot be cited until a Worker verifies them from fetched evidence.
    """

    case_id: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    assigned_task_key: str = ""
    field_requirement_ids: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


ArtifactRole = Literal[
    "source_document",
    "target_entity",
    "entity_container",
    "field_evidence",
    "navigation_only",
    "unknown",
]

CompletionContract = Literal[
    "discover_sources",
    "extract_entities",
    "verify_entity",
    "fill_fields",
    "resolve_conflict",
    "audit_scope",
    "synthesize",
]


class SearchFieldDefinition(StrictModel):
    """One LLM-designed field in the run-local relational search schema."""

    field_key: str
    label: str
    description: str = ""
    value_type: str = "text"
    required: bool = True
    temporal_scope: str = ""
    unit_or_format: str = ""
    aliases: list[str] = Field(default_factory=list)
    allowed_unknown_states: list[
        Literal["missing", "uncertain", "unreachable", "not_applicable"]
    ] = Field(default_factory=lambda: ["missing", "uncertain", "unreachable"])


class SearchTableDefinition(StrictModel):
    """A table whose row type and identity strategy are chosen by the Planner."""

    table_id: str
    row_entity_type: str
    description: str = ""
    primary_key_fields: list[str] = Field(default_factory=list)
    fields: list[SearchFieldDefinition] = Field(default_factory=list)


class SearchRelationDefinition(StrictModel):
    relation_id: str
    from_table: str
    from_field: str
    to_table: str
    to_field: str
    relation_type: str = "references"
    description: str = ""


class ScopeContract(StrictModel):
    """Auditable boundaries and LLM-proposed completion signals for a run."""

    target_description: str = ""
    inclusion_rules: list[str] = Field(default_factory=list)
    exclusion_rules: list[str] = Field(default_factory=list)
    time_boundary: str = ""
    geography: str = ""
    expected_count_hints: list[dict[str, Any]] = Field(default_factory=list)
    completion_signals: list[str] = Field(default_factory=list)


class SearchSchemaDefinition(StrictModel):
    schema_id: str
    version: int = Field(default=1, ge=1)
    status: Literal["provisional", "active", "superseded"] = "provisional"
    tables: list[SearchTableDefinition] = Field(default_factory=list)
    relations: list[SearchRelationDefinition] = Field(default_factory=list)
    scope_contract: ScopeContract = Field(default_factory=ScopeContract)
    rationale: str = ""
    created_by: Literal["planner", "focused_replan", "legacy_adapter"] = "planner"

    @model_validator(mode="after")
    def validate_relational_shape(self) -> "SearchSchemaDefinition":
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError("duplicate_search_table_id")
        tables = {table.table_id: table for table in self.tables}
        for table in self.tables:
            field_keys = [field.field_key for field in table.fields]
            if len(field_keys) != len(set(field_keys)):
                raise ValueError(f"duplicate_search_field_key:{table.table_id}")
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("duplicate_search_relation_id")
        for relation in self.relations:
            source = tables.get(relation.from_table)
            target = tables.get(relation.to_table)
            if source is None or target is None:
                raise ValueError(f"unknown_search_relation_table:{relation.relation_id}")
            source_fields = {field.field_key for field in source.fields} | {"entity_id"}
            target_fields = {field.field_key for field in target.fields} | {"entity_id"}
            if relation.from_field not in source_fields:
                raise ValueError(f"unknown_search_relation_field:{relation.relation_id}:from")
            if relation.to_field not in target_fields:
                raise ValueError(f"unknown_search_relation_field:{relation.relation_id}:to")
        return self


class SchemaRevision(StrictModel):
    revision_id: str
    schema_id: str
    from_version: int = Field(ge=1)
    to_version: int = Field(ge=1)
    reason: str
    changed_paths: list[str] = Field(default_factory=list)
    preserved_artifact_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class ResearchTask(StrictModel):
    task_id: str
    objective: str
    research_dimension: str
    # Planner-facing identity used to keep one Worker responsible for one
    # entity/object during a re-plan. Empty preserves old checkpoints.
    research_object: str = ""
    task_type: Literal["web", "direct_url", "counterevidence", "profile"] = "web"
    task_kind: Literal["gather", "analyze", "verify"] = "gather"
    depends_on: list[str] = Field(default_factory=list)
    consumes_requirement_ids: list[str] = Field(default_factory=list)
    produces_requirement_ids: list[str] = Field(default_factory=list)
    capability: Literal["web_research", "source_analysis", "claim_verification"] = (
        "web_research"
    )
    expected_value: float = Field(default=0.5, ge=0.0, le=1.0)
    source_origin: bool = False
    round_no: int = Field(default=1, ge=1, le=6)
    key_questions: list[str] = Field(default_factory=list)
    preferred_source_types: list[str] = Field(default_factory=list)
    verification_strategy: str = ""
    expected_output: str = ""
    candidate_cases: list[PlannerCandidateCase] = Field(default_factory=list)
    consumes_artifact_roles: list[ArtifactRole] = Field(default_factory=list)
    produces_artifact_roles: list[ArtifactRole] = Field(default_factory=list)
    target_entity_ids: list[str] = Field(default_factory=list)
    target_field_keys: list[str] = Field(default_factory=list)
    target_binding_ids: list[str] = Field(default_factory=list)
    completion_contract: CompletionContract = "discover_sources"
    retry_signature: str = ""
    priority_rationale: str = ""
    queries: list[str]
    requirement_ids: list[str] = Field(default_factory=list)
    priority: int = 1
    status: Literal["pending", "running", "completed", "failed", "superseded"] = "pending"
    attempt: int = 1
    budget: TaskBudget
    sufficiency: TaskSufficiency = Field(default_factory=TaskSufficiency)


class CandidateLead(StrictModel):
    """Non-citable lead extracted for downstream DAG tasks.

    Candidate leads guide exact-title/identifier verification. They are never
    admitted to the Writer claim ledger until a later task fetches and verifies
    the underlying source.
    """

    lead_id: str
    task_id: str
    title: str
    identifier: str = ""
    lead_type: Literal["paper", "report", "table", "dataset", "case", "other"] = "other"
    artifact_role: ArtifactRole = "unknown"
    target_entity_type: str = ""
    anchor_quote: str = ""
    extraction_strategy: Literal[
        "bibliography",
        "table_rows",
        "appendix",
        "named_items",
        "dataset_records",
        "follow_links",
        "none",
    ] = "none"
    origin_source_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    status: Literal["unverified", "verified", "rejected"] = "unverified"
    notes: str = ""


class SourceLead(StrictModel):
    """A source-shaped lead that cannot directly become a result entity."""

    lead_id: str
    task_id: str
    title: str
    identifier: str = ""
    document_type: str = "other"
    artifact_role: ArtifactRole = "unknown"
    target_entity_type: str = ""
    origin_source_id: str = ""
    anchor_quote: str = ""
    extraction_strategy: Literal[
        "bibliography",
        "table_rows",
        "appendix",
        "named_items",
        "dataset_records",
        "follow_links",
        "none",
    ] = "none"
    requirement_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    status: Literal["unverified", "usable", "rejected"] = "unverified"
    notes: str = ""


class EntityCandidate(StrictModel):
    """A run-local target candidate anchored to fetched evidence."""

    entity_id: str
    entity_type: str
    canonical_label: str
    aliases: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    origin_source_ids: list[str] = Field(default_factory=list)
    anchor_evidence_ids: list[str] = Field(default_factory=list)
    scope_status: Literal["in_scope", "uncertain", "out_of_scope"] = "uncertain"
    materialization_status: Literal[
        "discovered", "canonicalized", "materialized", "rejected"
    ] = "discovered"
    rationale: str = ""


class SearchHit(StrictModel):
    title: str
    url: str
    snippet: str = ""
    provider: str
    rank: int = 0
    score: float | None = None
    published_at: str | None = None


class CandidateSelection(BaseModel):
    """Worker decision over the complete coarse-filtered candidate inventory."""

    model_config = ConfigDict(extra="ignore")

    selected_candidate_ids: list[str] = Field(default_factory=list)
    next_action: Literal["fetch_selected", "search_more", "stop"] = "fetch_selected"
    rationale: str = ""

    @field_validator("selected_candidate_ids", mode="before")
    @classmethod
    def normalize_candidate_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


class PlannerTrace(StrictModel):
    run_id: str
    model_id: str
    round_no: int = 1
    raw_output: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    finish_reason: str | None = None
    reasoning_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


class SearchTrace(StrictModel):
    trace_id: str
    run_id: str
    task_id: str
    attempt: int
    query_id: str
    query: str
    requirement_ids: list[str] = Field(default_factory=list)
    results: list[SearchHit] = Field(default_factory=list)
    error: str | None = None


class FetchTrace(StrictModel):
    trace_id: str
    run_id: str
    task_id: str
    attempt: int
    query_id: str
    url: str
    requirement_ids: list[str] = Field(default_factory=list)
    provider: str
    status: Literal[
        "fetched",
        "failed",
        "no_evidence",
        "skipped_duplicate",
        "prefetched_unused",
    ]
    source_id: str | None = None
    accepted_claims: int = 0
    reuse_mode: Literal["none", "shared", "cached"] = "none"
    duration_seconds: float = 0.0
    network_duration_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    static_attempts: int = 0
    dynamic_attempted: bool = False
    retrieval_method: str | None = None
    http_status: int | None = None
    content_chars: int = 0
    error: str | None = None


class FetchedDocument(StrictModel):
    requested_url: str
    final_url: str
    title: str
    content: str
    content_type: str
    http_status: int
    fetched_at: str = Field(default_factory=utc_now)
    redirect_chain: list[str] = Field(default_factory=list)
    static_attempts: int = 0
    dynamic_attempted: bool = False
    retrieval_method: Literal[
        "direct",
        "alternative",
        "reader_fallback",
        "scrapling_dynamic",
        "academic_abstract",
        "academic_metadata",
    ] = "direct"


class ExtractedClaim(BaseModel):
    """Tolerant adapter for provider JSON; ledger validation remains strict."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(validation_alias=AliasChoices("text", "claim", "statement"))
    quote: str = Field(
        validation_alias=AliasChoices("quote", "source_quote", "evidence_quote", "evidence")
    )
    impact: Literal["high", "medium", "low"] = "medium"
    claim_type: str = "factual"
    requirement_ids: list[str] = Field(default_factory=list)
    entity_id: str = ""
    entity_label: str = ""
    entity_type: str = ""
    entity_aliases: list[str] = Field(default_factory=list)
    entity_identifiers: list[str] = Field(default_factory=list)
    entity_scope_status: Literal["in_scope", "uncertain", "out_of_scope"] = "uncertain"
    field_key: str = ""
    field_value: Any | None = None
    binding_status: Literal["supported", "conflicting", "uncertain"] = "supported"

    @field_validator("text", "quote", "claim_type")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("impact", mode="before")
    @classmethod
    def normalize_impact(cls, value: Any) -> str:
        normalized = str(value or "medium").strip().lower()
        return normalized if normalized in {"high", "medium", "low"} else "medium"

    @field_validator("entity_scope_status", mode="before")
    @classmethod
    def normalize_entity_scope_status(cls, value: Any) -> str:
        normalized = str(value or "uncertain").strip().lower()
        return (
            normalized
            if normalized in {"in_scope", "uncertain", "out_of_scope"}
            else "uncertain"
        )


class EvidenceExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        validation_alias=AliasChoices("claims", "factual_claims", "findings"),
    )
    knowledge_gaps: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("knowledge_gaps", "gaps", "uncertainties"),
    )
    artifact_role: ArtifactRole = "unknown"
    target_entity_type: str = ""
    extraction_strategy: Literal[
        "bibliography",
        "table_rows",
        "appendix",
        "named_items",
        "dataset_records",
        "follow_links",
        "none",
    ] = "none"

    @field_validator("extraction_strategy", mode="before")
    @classmethod
    def normalize_extraction_strategy(cls, value: Any) -> str:
        """Keep valid claims when a model swaps artifact_role and strategy.

        The strategy is descriptive metadata; an unknown or misplaced value
        must not invalidate otherwise usable evidence from the document.
        """

        normalized = str(value or "none").strip().casefold()
        allowed = {
            "bibliography",
            "table_rows",
            "appendix",
            "named_items",
            "dataset_records",
            "follow_links",
            "none",
        }
        return normalized if normalized in allowed else "none"

    @field_validator("claims", mode="before")
    @classmethod
    def normalize_claim_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return list(value.values())
        return value

    @field_validator("knowledge_gaps", mode="before")
    @classmethod
    def normalize_gap_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class SourceRecord(StrictModel):
    source_id: str
    source_kind: Literal["web_page", "search_snippet", "domain_record", "local_document"]
    title: str
    canonical_uri: str
    provider: str
    author_or_org: str | None = None
    published_at: str | None = None
    fetched_at: str
    fetch_status: Literal["full", "snippet_only", "failed"]
    http_status: int | None = None
    mime_type: str | None = None
    content_hash: str
    quality_tier: Literal[1, 2, 3, 4]
    quality_reason: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class SourceSnapshot(StrictModel):
    source_id: str
    content: str
    content_hash: str


class EvidenceRecord(StrictModel):
    evidence_id: str
    source_id: str
    quote: str
    quote_hash: str
    location: dict[str, str]
    requirement_ids: list[str] = Field(default_factory=list)


class ClaimRecord(StrictModel):
    claim_id: str
    text: str
    impact: Literal["high", "medium", "low"] = "medium"
    claim_type: str = "factual"
    status: Literal["supported", "contested", "needs_verification"] = "supported"
    requirement_ids: list[str] = Field(default_factory=list)


class ClaimEvidenceLink(StrictModel):
    link_id: str
    claim_id: str
    evidence_id: str
    relation: Literal["supports", "contradicts", "context"] = "supports"
    entailment_status: Literal["verified", "weak", "rejected"] = "verified"
    verification_method: Literal["deterministic", "reviewer"] = "deterministic"


class EvidenceBinding(StrictModel):
    """A field-level citation-matrix entry backed by a fetched EvidenceRecord."""

    binding_id: str
    evidence_id: str
    table_id: str = ""
    entity_id: str
    field_key: str
    requirement_id: str = ""
    proposed_value: Any | None = None
    status: Literal["supported", "conflicting", "uncertain", "rejected"] = "supported"
    rationale: str = ""


class EvidenceEdge(StrictModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation: Literal[
        "supports", "conflicts", "refines", "mentions", "contains", "same_as"
    ]
    evidence_id: str = ""
    rationale: str = ""


class CellCoverage(StrictModel):
    known_entity_count: int = Field(default=0, ge=0)
    materialized_cell_count: int = Field(default=0, ge=0)
    filled_cell_count: int = Field(default=0, ge=0)
    uncertain_cell_count: int = Field(default=0, ge=0)
    unreachable_cell_count: int = Field(default=0, ge=0)
    not_applicable_cell_count: int = Field(default=0, ge=0)
    missing_cell_count: int = Field(default=0, ge=0)
    ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ScopeCompleteness(StrictModel):
    status: Literal[
        "unknown",
        "open",
        "stabilizing",
        "provisionally_complete",
        "audited_complete",
        "unreachable",
    ] = "unknown"
    enumeration_source_ids: list[str] = Field(default_factory=list)
    discovered_entity_count: int = Field(default=0, ge=0)
    new_entities_by_window: list[int] = Field(default_factory=list)
    expected_count_hints: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_container_ids: list[str] = Field(default_factory=list)
    unresolved_alias_clusters: list[str] = Field(default_factory=list)
    audit_reason: str = ""


class CoverageChange(StrictModel):
    change_id: str
    entity_id: str = ""
    field_key: str = ""
    previous_status: str = ""
    new_status: Literal[
        "missing", "filled", "uncertain", "unreachable", "not_applicable"
    ]
    binding_id: str = ""
    reason: str = ""


class FailureRecord(StrictModel):
    failure_id: str
    scope_signature: str
    scope_type: Literal[
        "run", "task", "requirement", "query_angle", "url", "domain", "source_family"
    ] = "task"
    scope_value: str = ""
    task_id: str = ""
    failure_type: Literal[
        "query_no_gain",
        "duplicate_query",
        "url_unreachable",
        "transport_failure",
        "access_challenge",
        "dynamic_fetch_failure",
        "target_content_missing",
        "extract_no_binding",
        "source_format_mismatch",
        "skill_failure",
        "rejected_claim",
        "dropped_branch",
    ]
    action_signature: str
    correction: str = ""
    corrective_route: str = ""
    source_family: str = ""
    postmortem_eligible: bool = False
    recurrence_count: int = Field(default=1, ge=1)
    first_seen_at: str = Field(default_factory=utc_now)
    last_seen_at: str = Field(default_factory=utc_now)
    expires_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationDelta(StrictModel):
    """One replay-safe unit containing every state change from an observation."""

    delta_id: str
    task_id: str
    source_records: list[SourceRecord] = Field(default_factory=list)
    source_snapshots: list[SourceSnapshot] = Field(default_factory=list)
    source_leads: list[SourceLead] = Field(default_factory=list)
    entity_candidates: list[EntityCandidate] = Field(default_factory=list)
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    links: list[ClaimEvidenceLink] = Field(default_factory=list)
    bindings: list[EvidenceBinding] = Field(default_factory=list)
    graph_edges: list[EvidenceEdge] = Field(default_factory=list)
    failure_records: list[FailureRecord] = Field(default_factory=list)
    coverage_changes: list[CoverageChange] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class CitationRecord(StrictModel):
    citation_id: str
    section_id: str
    claim_id: str
    link_id: str


class UsageEvent(StrictModel):
    usage_event_id: str
    run_id: str
    task_id: str | None = None
    category: Literal["llm", "search", "fetch"]
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    fetches: int = 0
    estimated_cost_usd: float | None = None
    timestamp: str = Field(default_factory=utc_now)


class ResearchContribution(StrictModel):
    task_id: str
    agent_id: str
    model_id: str
    attempt: int
    status: Literal["completed", "failed", "timed_out", "cancelled"]
    stop_reason: str
    usable: bool
    result_brief: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    candidate_claim_ids: list[str] = Field(default_factory=list)
    candidate_lead_ids: list[str] = Field(default_factory=list)
    produced_artifact_roles: list[ArtifactRole] = Field(default_factory=list)
    produced_artifact_ids: list[str] = Field(default_factory=list)
    covered_query_ids: list[str] = Field(default_factory=list)
    query_count: int = 0
    successful_extractions: int = 0
    effective_source_count: int = 0
    independent_domain_count: int = 0
    evidence_count: int = 0
    sufficiency_passed: bool = False
    knowledge_gaps: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    started_at: str
    ended_at: str
    error: str | None = None


class ResearchGap(BaseModel):
    """Actionable feedback from the synthesizer to the next planner round."""

    model_config = ConfigDict(extra="ignore")

    gap_id: str = ""
    requirement_id: str = ""
    dimension: str
    missing_information: str
    why_needed: str = ""
    preferred_source_types: list[str] = Field(default_factory=list)
    verification_requirement: str = ""
    priority: Literal["high", "medium", "low"] = "medium"


class RequirementElementAssessment(BaseModel):
    """Semantic coverage of one user-visible requirement element."""

    model_config = ConfigDict(extra="ignore")

    element: str
    status: Literal["sufficient", "partial", "missing"]
    reason: str = ""
    supported_claim_ids: list[str] = Field(default_factory=list)


class CoverageAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dimension: str
    requirement_id: str = ""
    status: Literal["sufficient", "partial", "missing"]
    reason: str = ""
    supported_claim_ids: list[str] = Field(default_factory=list)
    required_element_assessments: list[RequirementElementAssessment] = Field(
        default_factory=list
    )


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_id: str
    status: Literal["supported", "partial", "conflicting", "uncertain"]
    reason: str = ""


class SynthesisDecision(BaseModel):
    """Semantic report-readiness decision; hard gates are applied in code."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["ready", "need_more", "partial"]
    answer_status: Literal["answerable", "not_yet_available", "insufficient"] = (
        "insufficient"
    )
    answer_summary: str = ""
    answer_supporting_claim_ids: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    coverage: list[CoverageAssessment] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    missing_information: list[ResearchGap] = Field(default_factory=list)


class SynthesisTrace(StrictModel):
    run_id: str
    round_no: int
    model_id: str
    decision: str
    raw_output: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    finish_reason: str | None = None
    reasoning_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    source_context_available: bool = False
    error: str | None = None


class WorkerTrace(StrictModel):
    run_id: str
    task_id: str
    round_no: int
    model_id: str
    phase: Literal["strategy", "reflection"]
    raw_output: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    finish_reason: str | None = None
    reasoning_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class ExtractionTrace(StrictModel):
    trace_id: str
    run_id: str
    task_id: str
    attempt: int
    source_uri: str
    chunk_index: int
    chunk_start: int
    chunk_end: int
    content_truncated: bool
    raw_output: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    finish_reason: str | None = None
    reasoning_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    completion_deadline_at: float | None = None
    selection_mode: Literal["legacy", "focused", "fallback_full"] = "legacy"
    focus_terms: list[str] = Field(default_factory=list)
    accepted_count: int = 0
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class WriterDraftParagraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    claim_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class WriterDraftSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    paragraphs: list[WriterDraftParagraph] = Field(default_factory=list)
    tables: list["WriterDraftTable"] = Field(default_factory=list)


class WriterDraftTableCell(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    cell_type: str = "factual"
    claim_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class WriterDraftTable(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[WriterDraftTableCell]] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)


class WriterMissingDimension(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dimension: str
    reason: str


class WriterDraft(BaseModel):
    """Structured, citation-addressable output accepted from the LLM writer."""

    model_config = ConfigDict(extra="ignore")

    sections: list[WriterDraftSection] = Field(default_factory=list)
    missing_dimensions: list[WriterMissingDimension] = Field(default_factory=list)


class WriterReviewIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location: str = "report"
    severity: str = "medium"
    issue_type: str = "grounding"
    message: str


class WriterReview(BaseModel):
    """Semantic review result; code only validates its shape and claim IDs."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["pass", "repair"]
    issues: list[WriterReviewIssue] = Field(default_factory=list)
    revised_draft: WriterDraft | None = None


class WriterTrace(StrictModel):
    run_id: str
    model_id: str
    writer_coverage_mode: Literal["full", "compact"] = "compact"
    raw_output: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""
    finish_reason: str | None = None
    reasoning_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    writer_duration_seconds: float = 0.0
    review_model_id: str | None = None
    review_output: dict[str, Any] = Field(default_factory=dict)
    review_input_tokens: int = 0
    review_output_tokens: int = 0
    reviewer_duration_seconds: float = 0.0
    rewrite_output: dict[str, Any] = Field(default_factory=dict)
    rewrite_input_tokens: int = 0
    rewrite_output_tokens: int = 0
    rewrite_duration_seconds: float = 0.0
    recheck_output: dict[str, Any] = Field(default_factory=dict)
    recheck_input_tokens: int = 0
    recheck_output_tokens: int = 0
    recheck_duration_seconds: float = 0.0
    review_applied: bool = False
    review_passed: bool = False
    review_error: str | None = None
    final_output: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    error: str | None = None


class OutputParagraph(StrictModel):
    paragraph_id: str
    text: str
    claim_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    paragraph_type: Literal[
        "factual", "transition", "user_constraint", "recommendation", "uncertainty"
    ] = "factual"


class OutputTableCell(StrictModel):
    cell_id: str = ""
    text: str
    cell_type: str = "factual"
    claim_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)


class OutputTable(StrictModel):
    table_id: str
    title: str
    columns: list[str]
    rows: list[list[OutputTableCell]]
    requirement_ids: list[str] = Field(default_factory=list)


class OutputSection(StrictModel):
    section_id: str
    title: str
    paragraphs: list[OutputParagraph]
    tables: list[OutputTable] = Field(default_factory=list)


class StructuredOutput(StrictModel):
    run_id: str
    mode: ResearchMode
    title: str
    direct_answer: str | None = None
    sections: list[OutputSection] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    run_status: RunStatus


class GateResult(StrictModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    grounded_citation_ids: list[str] = Field(default_factory=list)


class RunEvent(StrictModel):
    event_id: str
    run_id: str
    sequence: int
    timestamp: str = Field(default_factory=utc_now)
    event_type: str
    task_id: str | None = None
    agent_id: str | None = None
    status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ProgressEvent(StrictModel):
    event_id: str
    sequence: int = Field(ge=1)
    timestamp: str = Field(default_factory=utc_now)
    elapsed_seconds: float = Field(ge=0)
    event_type: str
    phase: str
    agent_role: str
    status: Literal["started", "progress", "completed", "warning", "failed", "heartbeat"]
    message: str
    visibility: Literal["quiet", "normal", "verbose"] = "normal"
    session_id: str | None = None
    turn_id: str | None = None
    run_id: str | None = None
    round_no: int | None = None
    task_id: str | None = None
    worker_id: str | None = None
    operation_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class RunResult(StrictModel):
    run_id: str
    status: RunStatus
    mode: ResearchMode
    output_path: str | None
    markdown: str
    sources: list[SourceRecord]
    claims: list[ClaimRecord]
    gate: GateResult
    usage: dict[str, int | float | None]
    artifact_dir: str
    audit_report_path: str | None = None
    audit_log_path: str | None = None


class ChatResult(StrictModel):
    user_id: str
    project_id: str
    session_id: str
    turn_id: str
    action: Literal["clarify", "research", "respond_from_context"]
    message: str
    run: RunResult | None = None
