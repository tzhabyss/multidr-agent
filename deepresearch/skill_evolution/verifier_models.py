"""Typed contracts for replay-backed CoEvoSkills verifier evolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import utc_now
from .models import EvolutionEvidenceReference, EvolutionSignalType


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OracleCheckStatus = Literal["pass", "fail", "unknown"]
OracleDecision = Literal["pass", "fail", "hold"]
ReplayLifecycleStatus = Literal["ready_for_shadow", "hold", "reject"]
VerifierAssertionKind = Literal[
    "required_hard_check",
    "metric_not_increase",
    "metric_not_decrease",
    "metric_strict_increase",
    "metric_strict_decrease",
]


class CandidateReplayObservation(_StrictModel):
    """A bounded Stable/Candidate comparison produced outside normal research."""

    schema_version: str = "1.0"
    replay_id: str
    candidate_id: str
    source_episode_id: str
    fixture_role: Literal["target", "guard"] = "target"
    stable_metrics: dict[str, float] = Field(default_factory=dict)
    candidate_metrics: dict[str, float] = Field(default_factory=dict)
    hard_checks: dict[str, bool] = Field(default_factory=dict)
    evidence_refs: list[EvolutionEvidenceReference] = Field(default_factory=list)
    external_writes_allowed: bool = False
    summary: str = ""
    created_at: str = Field(default_factory=utc_now)


class OracleCheck(_StrictModel):
    check_id: str
    status: OracleCheckStatus
    summary: str
    stable_value: float | bool | None = None
    candidate_value: float | bool | None = None


class ProjectOracleResult(_StrictModel):
    oracle_id: str = "deepresearch-project-oracle-v1"
    candidate_id: str
    replay_id: str
    verifier_suite_id: str
    verifier_suite_version: str
    decision: OracleDecision
    checks: list[OracleCheck] = Field(default_factory=list)
    summary: str = ""
    created_at: str = Field(default_factory=utc_now)


class VerifierGap(_StrictModel):
    """A surrogate-pass/project-oracle-fail event without hidden answer text."""

    schema_version: str = "1.0"
    gap_id: str
    candidate_id: str
    replay_id: str
    source_episode_id: str
    target_skill_id: str
    trigger_types: list[EvolutionSignalType]
    verifier_suite_id: str
    verifier_suite_version: str
    failed_check_ids: list[str]
    observable_failure: dict[str, object] = Field(default_factory=dict)
    evidence_refs: list[EvolutionEvidenceReference] = Field(default_factory=list)
    leak_safe_feedback: str
    status: Literal["proposed", "resolved"] = "proposed"
    created_at: str = Field(default_factory=utc_now)


class VerifierAssertion(_StrictModel):
    assertion_id: str
    kind: VerifierAssertionKind
    subject: str
    applies_to_skill_ids: list[str]
    trigger_types: list[EvolutionSignalType]
    tolerance: float = Field(default=0.0, ge=0.0)
    severity: Literal["warning", "error"] = "error"
    rationale: str
    provenance_gap_ids: list[str]


class VerifierSuite(_StrictModel):
    """Immutable suite; activation is represented by a separate release record."""

    schema_version: str = "1.0"
    verifier_suite_id: str
    version: str
    parent_version: str | None = None
    target_skill_id: str
    assertions: list[VerifierAssertion] = Field(default_factory=list)
    source_gap_ids: list[str] = Field(default_factory=list)
    status: Literal["builtin", "candidate"] = "candidate"
    generator_model_id: str = ""
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=utc_now)


class VerifierSuiteRelease(_StrictModel):
    schema_version: str = "1.0"
    release_id: str
    verifier_suite_id: str
    version: str
    parent_version: str | None = None
    target_skill_id: str
    human_review_approved: bool
    stable_skill_mutation_enabled: bool = False
    created_at: str = Field(default_factory=utc_now)


class CandidateReplayResult(_StrictModel):
    schema_version: str = "1.0"
    replay_id: str
    candidate_id: str
    source_episode_id: str
    status: ReplayLifecycleStatus
    observation: CandidateReplayObservation
    oracle_result: ProjectOracleResult
    verifier_gap_id: str = ""
    stable_skill_mutation_enabled: bool = False
    production_loading_enabled: bool = False
    reused_existing: bool = False
    created_at: str = Field(default_factory=utc_now)


class VerifierSuiteGenerationResult(_StrictModel):
    suite: VerifierSuite
    reused_existing: bool = False
    store_path: str = ""
