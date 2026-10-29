"""Explicit runtime configuration shared by all research agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .time_utils import resolve_timezone
from .tool_runtime import GovernanceConfig


class LLMOutputTokenConfig(BaseModel):
    """Completion-token allowances for each LLM responsibility."""

    model_config = ConfigDict(extra="forbid")

    planner: int = Field(ge=128)
    worker_strategy: int = Field(ge=128)
    worker_reflection: int = Field(ge=128)
    worker_agent_loop: int = Field(ge=128)
    evidence_extraction_min: int = Field(ge=128)
    evidence_extraction_max: int = Field(ge=128)
    synthesizer_assessment: int = Field(ge=128)
    report_writer: int = Field(ge=128)
    reviewer: int = Field(default=12000, ge=128)
    extraction_context: int = Field(ge=1024)
    clarifier: int = Field(default=1200, ge=128)
    session_response: int = Field(default=1600, ge=128)
    session_summary: int = Field(default=1200, ge=128)
    episode_summary: int = Field(default=1400, ge=128)

    @model_validator(mode="after")
    def validate_extraction_limits(self) -> "LLMOutputTokenConfig":
        if self.evidence_extraction_min > self.evidence_extraction_max:
            raise ValueError(
                "evidence_extraction_min must not exceed evidence_extraction_max"
            )
        if self.evidence_extraction_max >= self.extraction_context:
            raise ValueError(
                "evidence_extraction_max must be smaller than extraction_context"
            )
        return self


ModelTierName = Literal["lite", "default", "craft"]
ResearchModeName = Literal["quick", "standard", "deep", "max"]


class LLMProviderRuntimeConfig(BaseModel):
    """One OpenAI-compatible endpoint; secrets stay in environment variables."""

    model_config = ConfigDict(extra="forbid")

    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str = "https://api.openai.com/v1"
    request_timeout_seconds: float = Field(default=180.0, gt=0, le=1800)
    max_retries: int = Field(default=0, ge=0, le=5)


class ModelTierRuntimeConfig(BaseModel):
    """Logical model tier mapped to a provider and deployable model name."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "default"
    model: str | None = None
    model_env: str | None = None
    temperature: float = Field(default=0.1, ge=0, le=2)


class LLMRetryPolicyRuntimeConfig(BaseModel):
    """Application-level retry and tier failover for fragile final stages."""

    model_config = ConfigDict(extra="forbid")

    max_attempts_per_model: int = Field(default=2, ge=1, le=5)
    backoff_initial_seconds: float = Field(default=0.5, ge=0, le=30)
    backoff_multiplier: float = Field(default=2.0, ge=1, le=10)
    backoff_max_seconds: float = Field(default=4.0, ge=0, le=60)
    attempt_timeout_seconds: float | None = Field(default=None, gt=0, le=1800)
    recovery_strategy: Literal["sequential", "hedged"] = "sequential"
    hedged_primary_tiers: list[ModelTierName] = Field(
        default_factory=lambda: ["craft"]
    )
    fast_failure_window_seconds: float = Field(default=0, ge=0, le=60)
    fast_failure_retries: int = Field(default=0, ge=0, le=2)
    hedge_after_seconds: float | None = Field(default=None, gt=0, le=1800)
    fallback_attempt_timeout_seconds: float | None = Field(
        default=None, gt=0, le=1800
    )
    primary_result_grace_seconds: float = Field(default=0, ge=0, le=1800)
    fallback_tiers: list[ModelTierName] = Field(default_factory=list)

    @field_validator("fallback_tiers")
    @classmethod
    def deduplicate_fallback_tiers(
        cls, value: list[ModelTierName]
    ) -> list[ModelTierName]:
        return list(dict.fromkeys(value))

    @field_validator("hedged_primary_tiers")
    @classmethod
    def deduplicate_hedged_primary_tiers(
        cls, value: list[ModelTierName]
    ) -> list[ModelTierName]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_hedged_recovery(self) -> "LLMRetryPolicyRuntimeConfig":
        if self.recovery_strategy != "hedged":
            return self
        if self.hedge_after_seconds is None:
            raise ValueError("hedged recovery requires hedge_after_seconds")
        if self.attempt_timeout_seconds is None:
            raise ValueError("hedged recovery requires attempt_timeout_seconds")
        if self.hedge_after_seconds >= self.attempt_timeout_seconds:
            raise ValueError(
                "hedge_after_seconds must be smaller than attempt_timeout_seconds"
            )
        if not self.fallback_tiers:
            raise ValueError("hedged recovery requires at least one fallback tier")
        if not self.hedged_primary_tiers:
            raise ValueError("hedged recovery requires hedged_primary_tiers")
        return self

    def outer_timeout_seconds(self, *, cleanup_padding_seconds: float = 10.0) -> float | None:
        """Bound the complete recovery race, including a hedged fallback."""

        if self.attempt_timeout_seconds is None:
            return None
        if self.recovery_strategy == "hedged":
            fallback_timeout = (
                self.fallback_attempt_timeout_seconds
                or self.attempt_timeout_seconds
            )
            recovery_window = max(
                float(self.attempt_timeout_seconds),
                float(self.hedge_after_seconds or 0) + float(fallback_timeout),
            )
        else:
            recovery_window = (
                float(self.attempt_timeout_seconds)
                * self.max_attempts_per_model
                * max(1, 1 + len(self.fallback_tiers))
            )
        return recovery_window + max(0.0, float(cleanup_padding_seconds))


class LLMRecoveryRuntimeConfig(BaseModel):
    """Application-level recovery policies for latency-sensitive LLM stages."""

    model_config = ConfigDict(extra="forbid")

    clarifier: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            max_attempts_per_model=1,
            backoff_initial_seconds=0,
            attempt_timeout_seconds=90,
            recovery_strategy="hedged",
            hedged_primary_tiers=["lite", "default"],
            fast_failure_window_seconds=10,
            fast_failure_retries=1,
            hedge_after_seconds=30,
            fallback_attempt_timeout_seconds=60,
            primary_result_grace_seconds=5,
            fallback_tiers=["default", "craft"],
        )
    )
    synthesizer: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            max_attempts_per_model=1,
            backoff_initial_seconds=0,
            attempt_timeout_seconds=420,
            recovery_strategy="hedged",
            fast_failure_window_seconds=10,
            fast_failure_retries=1,
            hedge_after_seconds=180,
            fallback_attempt_timeout_seconds=240,
            primary_result_grace_seconds=20,
            fallback_tiers=["default"],
        )
    )
    planner: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            max_attempts_per_model=1,
            backoff_initial_seconds=0,
            attempt_timeout_seconds=1200,
            recovery_strategy="hedged",
            fast_failure_window_seconds=10,
            fast_failure_retries=1,
            hedge_after_seconds=800,
            fallback_attempt_timeout_seconds=400,
            # Keep the no-YAML/default construction path aligned with the
            # production runtime profile: if the fallback finishes first,
            # craft still owns the remainder of its full 20-minute window.
            primary_result_grace_seconds=400,
            fallback_tiers=["default"],
        )
    )
    worker_agent_loop: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            max_attempts_per_model=1,
            attempt_timeout_seconds=210,
            fallback_tiers=["craft"],
        )
    )
    report_writer: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            max_attempts_per_model=1,
            backoff_initial_seconds=0,
            attempt_timeout_seconds=1200,
            recovery_strategy="hedged",
            fast_failure_window_seconds=10,
            fast_failure_retries=1,
            hedge_after_seconds=700,
            fallback_attempt_timeout_seconds=500,
            primary_result_grace_seconds=10,
            fallback_tiers=["default"],
        )
    )
    reviewer: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            max_attempts_per_model=2,
            backoff_initial_seconds=0,
            attempt_timeout_seconds=120,
            fallback_tiers=["default"],
        )
    )
    judge: LLMRetryPolicyRuntimeConfig = Field(
        default_factory=lambda: LLMRetryPolicyRuntimeConfig(
            fallback_tiers=["craft", "lite"]
        )
    )

    def policy_for(self, role: str) -> LLMRetryPolicyRuntimeConfig | None:
        if role not in {
            "clarifier",
            "planner",
            "worker_agent_loop",
            "synthesizer",
            "report_writer",
            "reviewer",
            "judge",
        }:
            return None
        return getattr(self, role)


class StageModelRuntimeConfig(BaseModel):
    """Model tier selected for each LLM responsibility."""

    model_config = ConfigDict(extra="forbid")

    clarifier: ModelTierName = "lite"
    memory: ModelTierName = "lite"
    planner: ModelTierName = "default"
    worker_strategy: ModelTierName = "default"
    worker_agent_loop: ModelTierName = "default"
    evidence_extraction: ModelTierName = "default"
    worker_reflection: ModelTierName = "default"
    synthesizer: ModelTierName = "craft"
    report_writer: ModelTierName = "craft"
    reviewer: ModelTierName = "craft"


class ResearchModeRuntimeConfig(BaseModel):
    """Depth, concurrency, budget, and stage routing for one research mode."""

    model_config = ConfigDict(extra="forbid")

    stages: StageModelRuntimeConfig = Field(default_factory=StageModelRuntimeConfig)
    max_rounds: int = Field(default=3, ge=1, le=6)
    max_worker_iterations: int = Field(default=4, ge=1, le=12)
    max_workers: int = Field(default=8, ge=1, le=16)
    max_total_searches: int = Field(default=200, gt=0)
    max_total_fetches: int = Field(default=336, gt=0)
    max_wall_time_seconds: int = Field(default=1500, gt=0)
    research_soft_wall_time_seconds: int | None = Field(default=None, gt=0)
    writer_wall_time_seconds: int | None = Field(default=None, gt=0)
    reviewer_wall_time_seconds: int | None = Field(default=None, gt=0)
    writer_rewrite_wall_time_seconds: int | None = Field(default=None, gt=0)
    reviewer_recheck_wall_time_seconds: int | None = Field(default=None, gt=0)
    artifact_reserve_seconds: int = Field(default=0, ge=0)
    max_total_wall_time_seconds: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class ResearchModeOverrideRuntimeConfig(BaseModel):
    """Optional profile-specific overrides for one research mode."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int | None = Field(default=None, ge=1, le=6)
    max_worker_iterations: int | None = Field(default=None, ge=1, le=12)
    max_workers: int | None = Field(default=None, ge=1, le=16)
    max_total_searches: int | None = Field(default=None, gt=0)
    max_total_fetches: int | None = Field(default=None, gt=0)
    max_wall_time_seconds: int | None = Field(default=None, gt=0)
    research_soft_wall_time_seconds: int | None = Field(default=None, gt=0)
    writer_wall_time_seconds: int | None = Field(default=None, gt=0)
    reviewer_wall_time_seconds: int | None = Field(default=None, gt=0)
    writer_rewrite_wall_time_seconds: int | None = Field(default=None, gt=0)
    reviewer_recheck_wall_time_seconds: int | None = Field(default=None, gt=0)
    artifact_reserve_seconds: int | None = Field(default=None, ge=0)
    max_total_wall_time_seconds: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class AgentProfileRuntimeConfig(BaseModel):
    """Policy switches orthogonal to research depth and model routing."""

    model_config = ConfigDict(extra="forbid")

    semantic_citation_gate: Literal["off", "advisory", "repair", "strict"] = (
        "advisory"
    )
    requirement_matrix: bool = False
    # A delivery contract is a soft, user-visible coverage aid for requests
    # with explicit multi-part output structure.  It is intentionally separate
    # from the evaluation-only Requirement Matrix and never acts as a publish
    # gate when retrieval is incomplete.
    adaptive_delivery_contract: bool = True
    reviewer_mode: Literal["off", "on_demand", "always"] = "on_demand"
    exact_url_blocklist: bool = False
    enforce_candidate_quality_floor: bool = False
    max_tool_calls_per_iteration: int | None = Field(default=None, ge=1, le=16)
    min_observation_rounds: int = Field(default=0, ge=0, le=2)
    preserve_rich_markdown: bool = True
    include_audit_links: bool = True
    allow_derived_analysis: bool = False
    # Coverage remains available to the research control plane. The Writer
    # defaults to a compact user-facing context so field-level coverage state
    # cannot turn synthesis into checklist completion.
    writer_coverage_mode: Literal["full", "compact"] = "compact"
    # Skill evolution observation is profile-scoped so benchmark/evaluation
    # runs can remain immutable while normal research records learning signals.
    # ``inherit`` uses the global skill_evolution.mode value.
    skill_evolution_mode: Literal["inherit", "off", "observe"] = "inherit"
    skill_fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    skill_provider_modes: dict[str, Literal["fallback", "fanout"]] = Field(
        default_factory=dict
    )
    mode_overrides: dict[ResearchModeName, ResearchModeOverrideRuntimeConfig] = Field(
        default_factory=dict
    )

    @field_validator("skill_fallbacks")
    @classmethod
    def validate_skill_fallbacks(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for skill_name, fallback_names in value.items():
            parent = skill_name.strip()
            children = list(
                dict.fromkeys(item.strip() for item in fallback_names if item.strip())
            )
            if not parent or not children:
                raise ValueError(
                    "skill_fallbacks requires a skill name and at least one fallback"
                )
            normalized[parent] = children
        return normalized


P0AblationMode = Literal["legacy", "enhanced"]
P0FeatureMode = Literal["inherit", "legacy", "enhanced"]


class P0FeatureRuntimeConfig(BaseModel):
    """Per-feature overrides for SearchOS-inspired P0 ablations."""

    model_config = ConfigDict(extra="forbid")

    frontier: P0FeatureMode = "inherit"
    failure_memory: P0FeatureMode = "inherit"
    append_only_evidence: P0FeatureMode = "inherit"
    focused_page_processing: P0FeatureMode = "inherit"
    enhanced_loop_sensor: P0FeatureMode = "inherit"
    extraction_middleware: P0FeatureMode = "inherit"


class SkillLayerRuntimeConfig(BaseModel):
    """Visibility/injection policy for one progressive Skill layer."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    skills: list[str] = Field(default_factory=list)
    injection: Literal["none", "playbook", "selected", "tools"] = "none"


class SkillLayersRuntimeConfig(BaseModel):
    """Role-oriented Skill layers; empty orchestrator is intentional."""

    model_config = ConfigDict(extra="forbid")

    orchestrator: SkillLayerRuntimeConfig = Field(
        default_factory=lambda: SkillLayerRuntimeConfig(injection="none")
    )
    strategy: SkillLayerRuntimeConfig = Field(
        default_factory=lambda: SkillLayerRuntimeConfig(injection="selected")
    )
    access: SkillLayerRuntimeConfig = Field(
        default_factory=lambda: SkillLayerRuntimeConfig(injection="tools")
    )


class VerifierCoEvolutionRuntimeConfig(BaseModel):
    """Explicit, review-gated CoEvoSkills verifier evolution policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    generator_tier: ModelTierName = "craft"
    max_assertions_per_gap: int = Field(default=3, ge=1, le=5)
    generator_max_tokens: int = Field(default=2200, ge=256, le=12000)
    max_total_tokens_per_gap: int = Field(default=30000, ge=2000, le=100000)
    max_cost_regression_ratio: float = Field(default=0.2, ge=0.0, le=2.0)
    require_human_activation: bool = True
    stable_skill_auto_promotion: bool = False


class SkillEvolutionRuntimeConfig(BaseModel):
    """SkillX-Lite observation policy; phase one never loads Candidates."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "observe"] = "off"
    store_root: str = ".runtime/skill-evolution"
    duplicate_query_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    repeated_domain_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_signals_per_run: int = Field(default=100, ge=1, le=1000)
    candidate_generation: "CandidateGenerationRuntimeConfig" = Field(
        default_factory=lambda: CandidateGenerationRuntimeConfig()
    )
    verifier_coevolution: VerifierCoEvolutionRuntimeConfig = Field(
        default_factory=VerifierCoEvolutionRuntimeConfig
    )


class CandidateGenerationRuntimeConfig(BaseModel):
    """Bounded CoEvoSkills-Lite generation and verification policy."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    generator_tier: ModelTierName = "default"
    verifier_tier: ModelTierName = "craft"
    max_generator_verifier_rounds: int = Field(default=2, ge=1, le=2)
    max_similar_episodes: int = Field(default=3, ge=0, le=10)
    max_capsule_chars: int = Field(default=24000, ge=4000, le=100000)
    parent_skill_excerpt_chars: int = Field(default=8000, ge=500, le=30000)
    artifact_excerpt_chars: int = Field(default=4000, ge=500, le=20000)
    max_patch_operations: int = Field(default=3, ge=1, le=5)
    max_patch_chars: int = Field(default=1800, ge=100, le=10000)
    generator_max_tokens: int = Field(default=3000, ge=256, le=12000)
    verifier_max_tokens: int = Field(default=2200, ge=256, le=12000)
    max_total_tokens_per_candidate: int = Field(
        default=60000, ge=2000, le=300000
    )


class GraphEngineeringRuntimeConfig(BaseModel):
    """Deterministic control-plane policy for generated research DAGs."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    p0_ablation_mode: P0AblationMode = "legacy"
    p0_features: P0FeatureRuntimeConfig = Field(default_factory=P0FeatureRuntimeConfig)
    wave_scheduler_enabled: bool = True
    typed_search_state_enabled: bool = True
    continuous_frontier_enabled: bool = True
    role_projection_v2_enabled: bool = True
    evidence_middleware_enabled: bool = True
    sensor_middleware_enabled: bool = True
    failure_memory_enabled: bool = True
    atomic_observation_commits_enabled: bool = True
    wave_size: int = Field(default=16, ge=1, le=16)
    max_plan_tasks: int = Field(default=32, ge=1, le=32)
    max_dag_depth: int = Field(default=6, ge=1, le=12)
    max_zero_gain_waves: int = Field(default=2, ge=1, le=6)
    max_low_gain_waves: int = Field(default=2, ge=2, le=6)
    min_waves_before_low_gain_stop: int = Field(default=4, ge=2, le=8)
    low_gain_coverage_ratio: float = Field(default=0.75, ge=0.5, le=1.0)
    low_gain_requirement_fraction: float = Field(default=0.05, gt=0, le=0.25)
    low_gain_min_signals: int = Field(default=3, ge=1, le=4)
    max_worker_wall_time_seconds: int = Field(default=900, ge=30, le=3600)
    worker_finalization_reserve_seconds: int = Field(default=180, ge=10, le=600)
    worker_cleanup_reserve_seconds: int = Field(default=30, ge=5, le=120)
    extraction_soft_drain_seconds: int = Field(default=90, ge=0, le=300)
    research_finalization_reserve_seconds: int = Field(default=180, ge=10, le=600)
    fetch_batch_size: int = Field(default=8, ge=1, le=8)
    fetch_max_concurrency: int = Field(default=16, ge=1, le=64)
    fetch_max_concurrency_per_domain: int = Field(default=4, ge=1, le=16)
    fetch_prefetch_enabled: bool = True
    extraction_per_worker_concurrency: int = Field(default=2, ge=1, le=4)
    extraction_max_concurrency: int = Field(default=12, ge=1, le=32)
    candidate_selection_max_candidates: int = Field(default=50, ge=10, le=200)
    candidate_selection_timeout_seconds: int = Field(default=150, ge=30, le=600)
    min_worker_launch_seconds: int = Field(default=300, ge=60, le=1100)
    source_origin_first: bool = True
    task_dependencies_enabled: bool = True
    context_projection_enabled: bool = True
    deterministic_precheck_enabled: bool = True
    preserve_legacy_dispatch: bool = True
    # Skill loading can be ablated without changing the research DAG.  Legacy
    # reproduces the historical full-prompt behavior; layered uses metadata,
    # selected bodies, and lazy executors.
    skill_mode: Literal["legacy", "layered"] = "layered"

    def p0_feature_enabled(self, feature: str) -> bool:
        """Resolve one P0 feature while keeping the pre-change path selectable."""

        if feature not in P0FeatureRuntimeConfig.model_fields:
            raise ValueError(f"Unknown P0 feature: {feature}")
        override = getattr(self.p0_features, feature)
        selected = self.p0_ablation_mode if override == "inherit" else override
        return selected == "enhanced"

    def resolved_p0_features(self) -> dict[str, bool]:
        return {
            feature: self.p0_feature_enabled(feature)
            for feature in P0FeatureRuntimeConfig.model_fields
        }

    def p0_ablation_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.p0_ablation_mode,
            "features": self.resolved_p0_features(),
            "context_projection_changed": False,
        }

def _default_agent_profiles() -> dict[str, AgentProfileRuntimeConfig]:
    return {
        "general_research": AgentProfileRuntimeConfig(
            max_tool_calls_per_iteration=4,
            min_observation_rounds=1,
            allow_derived_analysis=True,
            skill_evolution_mode="observe",
            skill_fallbacks={
                "web-search": ["deepseek-web-search", "anysearch-search"]
            },
            skill_provider_modes={"web-search": "fanout"},
            mode_overrides={
                "standard": ResearchModeOverrideRuntimeConfig(
                    max_workers=8,
                    max_total_searches=100,
                    max_total_fetches=320,
                    max_wall_time_seconds=2400,
                    research_soft_wall_time_seconds=2100,
                    writer_wall_time_seconds=1220,
                    reviewer_wall_time_seconds=1020,
                    writer_rewrite_wall_time_seconds=1220,
                    reviewer_recheck_wall_time_seconds=500,
                    artifact_reserve_seconds=60,
                    max_total_wall_time_seconds=6420,
                )
            },
        ),
        "drb_evaluation": AgentProfileRuntimeConfig(
            semantic_citation_gate="advisory",
            requirement_matrix=True,
            adaptive_delivery_contract=True,
            reviewer_mode="always",
            exact_url_blocklist=True,
            enforce_candidate_quality_floor=True,
            max_tool_calls_per_iteration=4,
            min_observation_rounds=1,
            preserve_rich_markdown=True,
            include_audit_links=False,
            allow_derived_analysis=True,
            skill_evolution_mode="off",
            skill_fallbacks={
                "web-search": ["deepseek-web-search", "anysearch-search"]
            },
            skill_provider_modes={"web-search": "fanout"},
            mode_overrides={
                "standard": ResearchModeOverrideRuntimeConfig(
                    max_workers=8,
                    max_total_searches=100,
                    max_total_fetches=320,
                    max_wall_time_seconds=2400,
                    research_soft_wall_time_seconds=2100,
                    writer_wall_time_seconds=1220,
                    reviewer_wall_time_seconds=1020,
                    writer_rewrite_wall_time_seconds=1220,
                    reviewer_recheck_wall_time_seconds=500,
                    artifact_reserve_seconds=60,
                    max_total_wall_time_seconds=6420,
                )
            },
        ),
    }


class ToolPermissionRuntimeConfig(BaseModel):
    """Capability policy remains independent from model quality tiers."""

    model_config = ConfigDict(extra="forbid")

    executable_roles: list[str] = Field(default_factory=lambda: ["worker"])

    @field_validator("executable_roles")
    @classmethod
    def validate_roles(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not normalized:
            raise ValueError("executable_roles must contain at least one role")
        return normalized


def _default_model_tiers() -> dict[str, ModelTierRuntimeConfig]:
    return {
        "lite": ModelTierRuntimeConfig(model_env="DEEPRESEARCH_LITE_MODEL"),
        "default": ModelTierRuntimeConfig(model_env="DEEPRESEARCH_DEFAULT_MODEL"),
        "craft": ModelTierRuntimeConfig(model_env="DEEPRESEARCH_CRAFT_MODEL"),
    }


def _default_research_modes() -> dict[str, ResearchModeRuntimeConfig]:
    return {
        "quick": ResearchModeRuntimeConfig(
            stages=StageModelRuntimeConfig(
                planner="lite",
                evidence_extraction="lite",
                synthesizer="default",
                report_writer="default",
                reviewer="default",
            ),
            max_rounds=1,
            max_worker_iterations=2,
            max_workers=2,
            max_total_searches=16,
            max_total_fetches=32,
            max_wall_time_seconds=600,
        ),
        "standard": ResearchModeRuntimeConfig(
            max_rounds=4, max_workers=8, max_total_searches=100,
            max_total_fetches=320, max_wall_time_seconds=2400,
        ),
        "deep": ResearchModeRuntimeConfig(
            stages=StageModelRuntimeConfig(
                clarifier="default",
                planner="craft",
            ),
            max_rounds=4,
            max_worker_iterations=6,
            max_workers=10,
            max_total_searches=200,
            max_total_fetches=480,
            max_wall_time_seconds=3600,
        ),
        "max": ResearchModeRuntimeConfig(
            stages=StageModelRuntimeConfig(
                clarifier="default",
                memory="default",
                planner="craft",
                worker_strategy="craft",
                worker_agent_loop="craft",
                evidence_extraction="craft",
                worker_reflection="craft",
                synthesizer="craft",
                report_writer="craft",
            ),
            max_rounds=6,
            max_worker_iterations=8,
            max_workers=12,
            max_total_searches=320,
            max_total_fetches=640,
            max_wall_time_seconds=4800,
        ),
    }


class KnowledgeRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: Literal["sqlite", "milvus", "mongo_milvus"] = "milvus"
    path: str = ".runtime/knowledge.sqlite"
    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_database: str = "deepresearch_knowledge"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token_env: str = "MILVUS_TOKEN"
    milvus_namespace: str = "deepresearch_knowledge"
    milvus_lock_dir: str = ".runtime/knowledge-locks"
    parser: Literal["auto", "native", "mineru"] = "auto"
    ocr_backend: Literal["auto", "vision", "tesseract", "disabled"] = "auto"
    ocr_languages: str = "zh-Hans,en-US"
    ocr_timeout_seconds: float = Field(default=60, gt=0, le=300)
    parse_timeout_seconds: float = Field(default=300, gt=0, le=1800)
    parse_max_pages: int = Field(default=200, ge=1, le=2000)
    parse_allow_partial: bool = False
    embedding_model: Literal["intfloat/multilingual-e5-small", "BAAI/bge-small-zh-v1.5", "thenlper/gte-small-zh"] = "intfloat/multilingual-e5-small"
    embedding_revision: str | None = None
    embedding_device: Literal["cpu", "cuda", "mps"] | None = None
    collection: str = "library"
    # Opt-in cross-run source library, isolated by user + project request scope.
    persist_fetched_sources: bool = False
    contextual_index: bool = False
    document_context_path: str | None = None
    precontext_max_chars: int = Field(default=6000, ge=2000, le=24000)
    max_passages_per_source: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=3, ge=1, le=8)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    reranker_model: str | None = None
    reranker_revision: str | None = None
    reranker_device: Literal["cpu", "cuda", "mps"] | None = None
    reranker_max_length: int = Field(default=512, ge=128, le=8192)
    reranker_timeout_seconds: float = Field(default=5, gt=0, le=30)
    reranker_neighbor_chars: int = Field(default=0, ge=0, le=300)
    candidate_policy: Literal["rrf", "quota_equal", "quota_dense", "quota_lexical"] = "rrf"
    query_rewrite_enabled: bool = False
    query_rewrite_timeout_seconds: float = Field(default=15, gt=0, le=30)
    query_rewrite_fusion: Literal["rrf", "anchored_union"] = "rrf"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)

    llm_output_tokens: LLMOutputTokenConfig
    timezone: str = "Asia/Shanghai"
    knowledge: KnowledgeRuntimeConfig = Field(default_factory=KnowledgeRuntimeConfig)
    memory: "MemoryRuntimeConfig" = Field(default_factory=lambda: MemoryRuntimeConfig())
    mcp: "MCPRuntimeConfig" = Field(default_factory=lambda: MCPRuntimeConfig())
    default_mode: ResearchModeName = "standard"
    llm_providers: dict[str, LLMProviderRuntimeConfig] = Field(
        default_factory=lambda: {"default": LLMProviderRuntimeConfig()}
    )
    model_tiers: dict[str, ModelTierRuntimeConfig] = Field(
        default_factory=_default_model_tiers
    )
    llm_recovery: LLMRecoveryRuntimeConfig = Field(
        default_factory=LLMRecoveryRuntimeConfig
    )
    graph_engineering: GraphEngineeringRuntimeConfig = Field(
        default_factory=GraphEngineeringRuntimeConfig
    )
    skill_layers: SkillLayersRuntimeConfig = Field(
        default_factory=SkillLayersRuntimeConfig
    )
    skill_evolution: SkillEvolutionRuntimeConfig = Field(
        default_factory=SkillEvolutionRuntimeConfig
    )
    research_modes: dict[str, ResearchModeRuntimeConfig] = Field(
        default_factory=_default_research_modes
    )
    agent_profiles: dict[str, AgentProfileRuntimeConfig] = Field(
        default_factory=_default_agent_profiles
    )
    tool_permissions: ToolPermissionRuntimeConfig = Field(
        default_factory=ToolPermissionRuntimeConfig
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        resolve_timezone(value)
        return value

    @model_validator(mode="after")
    def validate_model_routing(self) -> "RuntimeConfig":
        missing_modes = {"quick", "standard", "deep", "max"} - set(
            self.research_modes
        )
        if missing_modes:
            raise ValueError(
                "research_modes is missing: " + ", ".join(sorted(missing_modes))
            )
        for tier_name in ("lite", "default", "craft"):
            tier = self.model_tiers.get(tier_name)
            if tier is None:
                raise ValueError(f"model_tiers is missing: {tier_name}")
            if tier.provider not in self.llm_providers:
                raise ValueError(
                    f"model tier {tier_name!r} references unknown provider "
                    f"{tier.provider!r}"
                )
        for role in (
            "planner",
            "worker_agent_loop",
            "synthesizer",
            "report_writer",
            "reviewer",
            "judge",
        ):
            policy = self.llm_recovery.policy_for(role)
            assert policy is not None
            unknown_fallbacks = set(policy.fallback_tiers) - set(self.model_tiers)
            if unknown_fallbacks:
                raise ValueError(
                    f"llm_recovery.{role} references unknown tiers: "
                    + ", ".join(sorted(unknown_fallbacks))
                )
        if "general_research" not in self.agent_profiles:
            raise ValueError("agent_profiles is missing: general_research")
        for profile_id, profile in self.agent_profiles.items():
            for mode_name, override in profile.mode_overrides.items():
                base = self.research_modes[mode_name]
                values = {
                    field_name: value
                    for field_name, value in override.model_dump().items()
                    if value is not None
                }
                selected = base.model_copy(update=values)
                soft = selected.research_soft_wall_time_seconds
                if soft is not None and soft > selected.max_wall_time_seconds:
                    raise ValueError(
                        f"{profile_id}.{mode_name} research soft deadline exceeds "
                        "the hard research deadline"
                    )
                timed_finalization = (
                    selected.writer_wall_time_seconds,
                    selected.reviewer_wall_time_seconds,
                    selected.writer_rewrite_wall_time_seconds,
                    selected.reviewer_recheck_wall_time_seconds,
                    selected.max_total_wall_time_seconds,
                )
                if any(value is not None for value in timed_finalization) and not all(
                    value is not None for value in timed_finalization
                ):
                    raise ValueError(
                        f"{profile_id}.{mode_name} must configure Writer, Reviewer, "
                        "Writer rewrite, Reviewer recheck, and total wall times together"
                    )
                if selected.max_total_wall_time_seconds is not None:
                    required = (
                        selected.max_wall_time_seconds
                        + int(selected.writer_wall_time_seconds or 0)
                        + int(selected.reviewer_wall_time_seconds or 0)
                        + int(selected.writer_rewrite_wall_time_seconds or 0)
                        + int(selected.reviewer_recheck_wall_time_seconds or 0)
                        + selected.artifact_reserve_seconds
                    )
                    if required > selected.max_total_wall_time_seconds:
                        raise ValueError(
                            f"{profile_id}.{mode_name} stage deadlines require "
                            f"{required}s but total wall time is "
                            f"{selected.max_total_wall_time_seconds}s"
                        )
        return self

    @staticmethod
    def normalize_mode(mode: str | None) -> ResearchModeName:
        value = str(mode or "standard").strip().lower()
        aliases = {
            "research": "standard",
            "deep_search": "standard",
            "deep_report": "standard",
        }
        value = aliases.get(value, value)
        if value not in {"quick", "standard", "deep", "max"}:
            raise ValueError(f"Unknown research mode: {mode}")
        return value  # type: ignore[return-value]

    def mode_config(self, mode: str | None = None) -> ResearchModeRuntimeConfig:
        return self.research_modes[self.normalize_mode(mode or self.default_mode)]

    def stage_tier(self, stage: str, mode: str | None = None) -> str:
        stages = self.mode_config(mode).stages
        if stage not in StageModelRuntimeConfig.model_fields:
            raise ValueError(f"Unknown LLM stage: {stage}")
        return str(getattr(stages, stage))

    def profile_config(self, profile_id: str | None = None) -> AgentProfileRuntimeConfig:
        selected = str(profile_id or "general_research").strip()
        try:
            return self.agent_profiles[selected]
        except KeyError as exc:
            raise ValueError(f"Unknown agent profile: {selected}") from exc

    def effective_skill_evolution_mode(
        self, profile_id: str | None = None
    ) -> Literal["off", "observe"]:
        """Resolve global Skill observation policy with a profile override."""

        profile_mode = self.profile_config(profile_id).skill_evolution_mode
        if profile_mode == "inherit":
            return self.skill_evolution.mode
        return profile_mode

    def effective_mode_config(
        self,
        mode: str | None = None,
        profile_id: str | None = None,
    ) -> ResearchModeRuntimeConfig:
        """Apply the selected profile's sparse override to mode defaults."""

        mode_name = self.normalize_mode(mode or self.default_mode)
        selected = self.mode_config(mode_name)
        override = self.profile_config(profile_id).mode_overrides.get(mode_name)
        if override is None:
            return selected
        updates = {
            field_name: value
            for field_name, value in override.model_dump().items()
            if value is not None
        }
        return selected.model_copy(update=updates)

    def resolve_research_options(
        self,
        options: Any,
        mode: str | None = None,
        profile_id: str | None = None,
    ) -> dict[str, int]:
        """Merge explicit request overrides onto the selected mode defaults."""

        selected = self.effective_mode_config(mode, profile_id)
        explicit = getattr(options, "model_fields_set", set())
        return {
            "max_rounds": (
                options.max_rounds if "max_rounds" in explicit else selected.max_rounds
            ),
            "max_worker_iterations": (
                options.max_worker_iterations
                if "max_worker_iterations" in explicit
                else selected.max_worker_iterations
            ),
            "max_workers": (
                options.max_workers if "max_workers" in explicit else selected.max_workers
            ),
        }


class MemoryRuntimeConfig(BaseModel):
    """Session and cross-session memory behavior.

    The project namespace is intentionally fixed for v1. Redis remains an
    optional session accelerator; durable facts, revisions, capture outboxes,
    and retrieval indexes are owned by the local autonomous memory subsystem.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    project_id: str = "deepresearch-default"
    default_user_id: str = "local-user"
    session_backend: str = "local"
    redis_url_env: str = "DEEPRESEARCH_REDIS_URL"
    local_db_path: str = ".runtime/memory.sqlite"
    session_ttl_hours: int = Field(default=168, ge=1)
    recent_turns: int = Field(default=8, ge=2, le=30)
    context_window_tokens: int = Field(default=100000, ge=10000, le=2000000)
    # Start durable session compaction at 60% of the default 100k window.
    # Explicit lower values remain useful for tests and small deployments.
    summary_trigger_tokens: int = Field(default=60000, ge=1000)
    compaction_light_ratio: float = Field(default=0.60, gt=0.0, lt=1.0)
    compaction_query_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
    compaction_summary_ratio: float = Field(default=0.88, gt=0.0, lt=1.0)
    compaction_emergency_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
    automatic_capture_enabled: bool = True
    fact_top_k_clarifier: int = Field(default=3, ge=0, le=20)
    fact_top_k_planner: int = Field(default=5, ge=0, le=20)
    fact_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    capture_batch_size: int = Field(default=20, ge=1, le=200)
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimensions: int = Field(default=384, ge=32, le=4096)
    embedding_query_prefix: str = "query: "
    embedding_document_prefix: str = "passage: "
    embedding_fallback_enabled: bool = True
    semantic_candidate_limit: int = Field(default=50, ge=5, le=1000)
    lexical_candidate_limit: int = Field(default=50, ge=5, le=1000)
    semantic_dedup_threshold: float = Field(default=0.92, ge=0.5, le=1.0)
    segment_turns: int = Field(default=8, ge=2, le=30)
    rolling_segment_limit: int = Field(default=12, ge=2, le=100)
    hot_message_max_chars: int = Field(default=6000, ge=1000, le=50000)
    clarifier_memory_budget_tokens: int = Field(default=6000, ge=500, le=50000)
    planner_memory_budget_tokens: int = Field(default=16000, ge=1000, le=80000)
    worker_memory_budget_tokens: int = Field(default=8000, ge=500, le=50000)
    projection_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    projection_sentence_limit: int = Field(default=8, ge=1, le=50)
    mandatory_fact_limit: int = Field(default=20, ge=0, le=100)
    max_facts_per_scope: int = Field(default=500, ge=20, le=100000)
    default_fact_valid_days: int = Field(default=365, ge=1, le=36500)
    lifecycle_enabled: bool = True

    @model_validator(mode="after")
    def validate_compaction_thresholds(self) -> "MemoryRuntimeConfig":
        ratios = (
            self.compaction_light_ratio,
            self.compaction_query_ratio,
            self.compaction_summary_ratio,
            self.compaction_emergency_ratio,
        )
        if tuple(sorted(ratios)) != ratios or len(set(ratios)) != len(ratios):
            raise ValueError("memory compaction ratios must be strictly increasing")
        if self.summary_trigger_tokens > self.context_window_tokens:
            raise ValueError(
                "summary_trigger_tokens must not exceed context_window_tokens"
            )
        return self


class MCPToolRuntimeConfig(BaseModel):
    """Permission and runtime policy for one explicitly allowlisted MCP tool."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    skill_id: str | None = None
    description: str | None = None
    roles: list[str] = Field(default_factory=lambda: ["worker"])
    profiles: list[str] = Field(default_factory=lambda: ["general_research"])
    tags: list[str] = Field(default_factory=list)
    result_kind: Literal["search_hits", "observation"] = "observation"
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)


class MCPServerRuntimeConfig(BaseModel):
    """Connection and allowlist configuration for one MCP server."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    enabled_env: str | None = None
    required: bool = False
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    env_files: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    connect_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    tools: dict[str, MCPToolRuntimeConfig] = Field(default_factory=dict)

    def runtime_enabled(self) -> bool:
        if not self.enabled:
            return False
        if not self.enabled_env:
            return True
        value = os.getenv(self.enabled_env, "").strip().lower()
        return value not in {"", "0", "false", "no", "off"}

    @model_validator(mode="after")
    def validate_transport_configuration(self) -> "MCPServerRuntimeConfig":
        if not self.enabled:
            return self
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers require command")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable_http MCP servers require url")
        return self


class MCPRuntimeConfig(BaseModel):
    """MCP remains disabled until a server and its tool allowlist are configured."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    servers: dict[str, MCPServerRuntimeConfig] = Field(default_factory=dict)


DEFAULT_RUNTIME_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "runtime.yaml"
)


def resolve_runtime_config_path(path: str | Path | None = None) -> Path:
    configured = path or os.getenv("DEEPRESEARCH_CONFIG")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_RUNTIME_CONFIG_PATH


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    config_path = resolve_runtime_config_path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read runtime config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in runtime config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Runtime config {config_path} must contain a YAML object")
    return RuntimeConfig.model_validate(raw)
