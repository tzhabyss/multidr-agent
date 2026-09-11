"""Typed contracts for SkillX-Lite signals and Evolution Episodes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import utc_now


SkillCapabilityGranularity = Literal["strategic", "functional", "atomic"]
CandidatePolicy = Literal[
    "observe_only",
    "episode_requires_review",
    "automatic_candidate_allowed",
]
EvolutionSignalType = Literal[
    "consecutive_no_evidence_gain",
    "duplicate_query",
    "repeated_source_domain",
    "source_family_repeated_failure",
    "fetch_route_failure",
    "extraction_without_binding",
    "skill_execution_failure",
    "weak_or_rejected_claim",
]
PatchOperationKind = Literal["add", "delete", "replace"]
CandidateLifecycleStatus = Literal[
    "draft",
    "revise",
    "accept_for_replay",
    "hold",
    "reject",
]
VerifierDecision = Literal["accept_for_replay", "revise", "hold", "reject"]


def candidate_policy_for(
    granularity: SkillCapabilityGranularity,
) -> CandidatePolicy:
    """Return the phase-one boundary; this does not create a Candidate."""

    if granularity == "atomic":
        return "automatic_candidate_allowed"
    if granularity == "functional":
        return "episode_requires_review"
    return "observe_only"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvolutionEvidenceReference(_StrictModel):
    artifact_path: str
    record_type: str
    record_id: str = ""


class EvolutionSignal(_StrictModel):
    schema_version: str = "1.0"
    signal_id: str
    signal_type: EvolutionSignalType
    run_id: str
    task_id: str = ""
    window_id: str = ""
    target_skill_id: str
    target_skill_version: str
    runtime_layer: Literal["orchestrator", "strategy", "access"]
    capability_granularity: SkillCapabilityGranularity
    candidate_policy: CandidatePolicy
    scope_type: Literal[
        "run", "task", "query", "url", "domain", "source_family", "claim"
    ] = "task"
    scope_value: str = ""
    severity: Literal["info", "warning", "error"] = "warning"
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    context: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvolutionEvidenceReference] = Field(default_factory=list)
    observed_at: str = Field(default_factory=utc_now)


class EvolutionEpisode(_StrictModel):
    schema_version: str = "1.0"
    episode_id: str
    run_id: str
    task_id: str = ""
    target_skill_id: str
    target_skill_version: str
    runtime_layer: Literal["orchestrator", "strategy", "access"]
    capability_granularity: SkillCapabilityGranularity
    candidate_policy: CandidatePolicy
    trigger_signal_ids: list[str]
    trigger_types: list[EvolutionSignalType]
    scope: dict[str, str] = Field(default_factory=dict)
    local_context: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    local_result: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvolutionEvidenceReference] = Field(default_factory=list)
    replay_spec: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str = Field(default_factory=utc_now)


class CandidateEvidenceExcerpt(_StrictModel):
    artifact_path: str
    record_type: str
    record_id: str = ""
    content: Any
    truncated: bool = False


class CandidateInputCapsule(_StrictModel):
    """Bounded phase-two input; never contains the whole conversation."""

    schema_version: str = "1.0"
    capsule_id: str
    source_episode_id: str
    run_id: str
    task_id: str = ""
    primary_trigger_type: EvolutionSignalType
    trigger_summary: str
    target_skill_id: str
    parent_skill_version: str
    runtime_layer: Literal["orchestrator", "strategy", "access"]
    capability_granularity: SkillCapabilityGranularity
    candidate_policy: CandidatePolicy
    scope: dict[str, str] = Field(default_factory=dict)
    local_context: dict[str, Any] = Field(default_factory=dict)
    local_actions: list[dict[str, Any]] = Field(default_factory=list)
    local_result: dict[str, Any] = Field(default_factory=dict)
    expected_local_outcomes: list[str] = Field(default_factory=list)
    parent_skill_excerpt: str
    parent_skill_source_path: str
    evidence_excerpts: list[CandidateEvidenceExcerpt] = Field(default_factory=list)
    similar_episodes: list[dict[str, Any]] = Field(default_factory=list)
    allowed_boundary: dict[str, Any] = Field(default_factory=dict)
    total_chars: int = Field(ge=0)
    truncated: bool = False


class PatchOperation(_StrictModel):
    operation_id: str
    kind: PatchOperationKind
    target_section: str
    anchor: str = ""
    old_text: str = ""
    new_text: str = ""
    rationale: str


class CandidatePatch(_StrictModel):
    schema_version: str = "1.0"
    candidate_id: str
    target_skill_id: str
    parent_skill_version: str
    capability_granularity: SkillCapabilityGranularity
    trigger_conditions: list[str]
    applicability_preconditions: list[str]
    operations: list[PatchOperation]
    expected_improvements: list[str]
    risks: list[str]
    source_episode_ids: list[str]
    requested_tools: list[str] = Field(default_factory=list)
    requested_permissions: list[str] = Field(default_factory=list)
    gate_bypasses: list[str] = Field(default_factory=list)
    patch_fingerprint: str = ""


class VerifierIssue(_StrictModel):
    code: str
    severity: Literal["warning", "error"]
    message: str
    suggested_fix: str = ""


class CandidateVerification(_StrictModel):
    verifier_id: str = "coevoskills-lite-fixed-v1"
    decision: VerifierDecision
    summary: str
    issues: list[VerifierIssue] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class CandidateRevision(_StrictModel):
    round_no: int = Field(ge=1)
    patch: CandidatePatch
    deterministic_issues: list[VerifierIssue] = Field(default_factory=list)
    verification: CandidateVerification | None = None
    changed_from_previous: bool = True
    generator_input_tokens: int = Field(default=0, ge=0)
    generator_output_tokens: int = Field(default=0, ge=0)


class CandidateRecord(_StrictModel):
    """Immutable audit record for one Episode's phase-two lifecycle."""

    schema_version: str = "1.0"
    candidate_id: str
    parent_candidate_id: str = ""
    source_episode_id: str
    source_verifier_gap_ids: list[str] = Field(default_factory=list)
    run_id: str
    target_skill_id: str
    parent_skill_version: str
    capability_granularity: SkillCapabilityGranularity
    status: CandidateLifecycleStatus
    capsule: CandidateInputCapsule | None = None
    revisions: list[CandidateRevision] = Field(default_factory=list)
    final_patch: CandidatePatch | None = None
    final_reasons: list[str] = Field(default_factory=list)
    generator_model_id: str = ""
    verifier_model_id: str = ""
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    stable_skill_mutation_enabled: bool = False
    production_loading_enabled: bool = False
    created_at: str = Field(default_factory=utc_now)


class CandidateGenerationResult(_StrictModel):
    record: CandidateRecord
    reused_existing: bool = False
    store_path: str = ""
