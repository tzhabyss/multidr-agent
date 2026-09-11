from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deepresearch.agents import Planner, Synthesizer
from deepresearch.llm import LLMResponse
from deepresearch.model_routing import ModelRouter
from deepresearch.runtime_config import RuntimeConfig, load_runtime_config
from deepresearch.skill_loader import discover_skills
from deepresearch.schemas import (
    ResearchMode,
    ResearchOptions,
    ResearchRequest,
    ResearchScope,
    TaskBudget,
)
from deepresearch.service import ResearchService

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


def _config(**overrides: int) -> RuntimeConfig:
    values = {
        "planner": 2200,
        "worker_strategy": 1000,
        "worker_reflection": 900,
        "worker_agent_loop": 1000,
        "evidence_extraction_min": 256,
        "evidence_extraction_max": 1200,
        "synthesizer_assessment": 4000,
        "report_writer": 3000,
        "extraction_context": 24000,
    }
    values.update(overrides)
    return RuntimeConfig.model_validate({"llm_output_tokens": values})


def test_default_runtime_config_exposes_all_llm_output_limits():
    config = load_runtime_config()

    assert config.llm_output_tokens.planner == 40000
    assert config.llm_output_tokens.synthesizer_assessment == 8192
    assert config.llm_output_tokens.report_writer == 30000
    assert config.llm_output_tokens.reviewer == 12000
    assert config.llm_output_tokens.evidence_extraction_max == 8000
    assert config.llm_output_tokens.clarifier == 1200
    assert config.memory.project_id == "deepresearch-default"
    assert config.mcp.enabled is True
    deepseek_mcp = config.mcp.servers["deepseek-web-search"]
    assert deepseek_mcp.enabled_env is None
    assert deepseek_mcp.env["DEEPSEEK_API_KEY"] == "${DEEPSEEK_API_KEY}"
    assert deepseek_mcp.tools["web_search"].profiles == [
        "general_research",
        "drb_evaluation",
    ]
    assert config.default_mode == "standard"
    assert config.llm_providers["default"].request_timeout_seconds == 1500
    assert config.llm_providers["default"].max_retries == 0
    assert config.llm_recovery.planner.attempt_timeout_seconds == 1200
    assert config.llm_recovery.planner.max_attempts_per_model == 1
    assert config.llm_recovery.worker_agent_loop.attempt_timeout_seconds == 210
    assert config.llm_recovery.worker_agent_loop.max_attempts_per_model == 1
    assert config.llm_recovery.worker_agent_loop.fallback_tiers == ["craft"]
    assert config.llm_recovery.report_writer.max_attempts_per_model == 1
    assert config.llm_recovery.report_writer.backoff_initial_seconds == 0
    assert config.llm_recovery.report_writer.recovery_strategy == "hedged"
    assert config.llm_recovery.report_writer.hedged_primary_tiers == ["craft"]
    assert config.llm_recovery.report_writer.fast_failure_window_seconds == 10
    assert config.llm_recovery.report_writer.fast_failure_retries == 1
    assert config.llm_recovery.report_writer.hedge_after_seconds == 700
    assert config.llm_recovery.report_writer.attempt_timeout_seconds == 1200
    assert config.llm_recovery.report_writer.fallback_attempt_timeout_seconds == 500
    assert config.llm_recovery.report_writer.primary_result_grace_seconds == 10
    assert config.llm_recovery.report_writer.fallback_tiers == ["default"]
    assert config.llm_recovery.planner.recovery_strategy == "hedged"
    assert config.llm_recovery.planner.hedged_primary_tiers == ["craft"]
    assert config.llm_recovery.planner.fast_failure_window_seconds == 10
    assert config.llm_recovery.planner.fast_failure_retries == 1
    assert config.llm_recovery.planner.hedge_after_seconds == 800
    assert config.llm_recovery.planner.fallback_attempt_timeout_seconds == 400
    assert config.llm_recovery.planner.primary_result_grace_seconds == 400
    assert config.llm_recovery.planner.fallback_tiers == ["default"]
    assert config.llm_recovery.planner.outer_timeout_seconds() == 1210
    assert config.llm_recovery.reviewer.max_attempts_per_model == 2
    assert config.llm_recovery.reviewer.attempt_timeout_seconds == 120
    assert config.llm_recovery.reviewer.fallback_tiers == ["default"]
    assert config.llm_recovery.clarifier.recovery_strategy == "hedged"
    assert config.llm_recovery.clarifier.hedge_after_seconds == 30
    assert config.llm_recovery.clarifier.outer_timeout_seconds() == 100
    assert config.llm_recovery.synthesizer.recovery_strategy == "hedged"
    assert config.llm_recovery.synthesizer.hedge_after_seconds == 180
    assert config.llm_recovery.synthesizer.outer_timeout_seconds() == 430
    assert config.graph_engineering.max_low_gain_waves == 2
    assert config.graph_engineering.min_waves_before_low_gain_stop == 4
    assert config.graph_engineering.low_gain_coverage_ratio == 0.75
    assert config.graph_engineering.low_gain_min_signals == 3
    assert config.graph_engineering.fetch_batch_size == 8
    assert config.graph_engineering.fetch_max_concurrency == 16
    assert config.graph_engineering.fetch_max_concurrency_per_domain == 4
    assert config.graph_engineering.fetch_prefetch_enabled is True
    assert config.graph_engineering.extraction_per_worker_concurrency == 2
    assert config.graph_engineering.extraction_max_concurrency == 12
    assert config.graph_engineering.extraction_soft_drain_seconds == 90
    assert not config.graph_engineering.p0_feature_enabled(
        "focused_page_processing"
    )
    assert config.mode_config("quick").max_workers == 2
    assert config.mode_config("standard").max_worker_iterations == 4
    assert config.mode_config("deep").max_rounds == 4
    assert config.graph_engineering.wave_size == 16
    assert config.graph_engineering.max_plan_tasks == 32
    assert config.mode_config("max").max_total_fetches == 480
    assert config.stage_tier("planner", "quick") == "lite"
    assert config.stage_tier("synthesizer", "standard") == "craft"
    assert config.tool_permissions.executable_roles == ["worker"]
    general_profile = config.profile_config("general_research")
    assert general_profile.requirement_matrix is False
    assert general_profile.adaptive_delivery_contract is True
    assert general_profile.reviewer_mode == "on_demand"
    assert general_profile.exact_url_blocklist is False
    assert general_profile.enforce_candidate_quality_floor is False
    assert general_profile.semantic_citation_gate == "advisory"
    assert general_profile.writer_coverage_mode == "compact"
    assert general_profile.skill_evolution_mode == "observe"
    assert config.effective_skill_evolution_mode("general_research") == "observe"
    assert general_profile.max_tool_calls_per_iteration == 4
    assert general_profile.min_observation_rounds == 1
    assert general_profile.skill_fallbacks == {
        "web-search": ["deepseek-web-search", "anysearch-search"]
    }
    assert general_profile.skill_provider_modes == {"web-search": "fanout"}
    evaluation = config.profile_config("drb_evaluation")
    assert evaluation.semantic_citation_gate == "advisory"
    assert evaluation.requirement_matrix is True
    assert evaluation.adaptive_delivery_contract is True
    assert evaluation.reviewer_mode == "always"
    assert evaluation.enforce_candidate_quality_floor is True
    assert evaluation.max_tool_calls_per_iteration == 4
    assert evaluation.include_audit_links is False
    assert evaluation.writer_coverage_mode == "compact"
    assert evaluation.skill_evolution_mode == "off"
    assert config.effective_skill_evolution_mode("drb_evaluation") == "off"
    assert evaluation.skill_fallbacks == {
        "web-search": ["deepseek-web-search", "anysearch-search"]
    }
    assert evaluation.skill_provider_modes == {"web-search": "fanout"}
    evaluation_standard = config.effective_mode_config(
        "standard", "drb_evaluation"
    )
    assert evaluation_standard.max_workers == 8
    assert evaluation_standard.max_total_searches == 200
    assert evaluation_standard.max_total_fetches == 336
    assert evaluation_standard.research_soft_wall_time_seconds == 3780
    assert evaluation_standard.max_wall_time_seconds == 4080
    assert evaluation_standard.writer_wall_time_seconds == 1220
    assert evaluation_standard.reviewer_wall_time_seconds == 1020
    assert evaluation_standard.writer_rewrite_wall_time_seconds == 1220
    assert evaluation_standard.reviewer_recheck_wall_time_seconds == 500
    assert evaluation_standard.artifact_reserve_seconds == 60
    assert evaluation_standard.max_total_wall_time_seconds == 8100
    assert evaluation_standard.max_rounds == 4
    assert evaluation_standard.max_worker_iterations == 4


def test_mode_defaults_only_fill_options_not_explicitly_set():
    config = _config()

    defaults = config.resolve_research_options(ResearchOptions(), "deep")
    assert defaults == {
        "max_rounds": 4,
        "max_worker_iterations": 6,
        "max_workers": 12,
    }

    explicit = config.resolve_research_options(
        ResearchOptions(max_rounds=2), "deep"
    )
    assert explicit == {
        "max_rounds": 2,
        "max_worker_iterations": 6,
        "max_workers": 12,
    }


def test_standard_capacity_is_shared_and_explicit_options_win(tmp_path):
    config = load_runtime_config()

    general = config.effective_mode_config("standard", "general_research")
    evaluation = config.effective_mode_config("standard", "drb_evaluation")
    assert (
        general.max_workers,
        general.max_total_searches,
        general.max_total_fetches,
        general.max_wall_time_seconds,
    ) == (8, 200, 336, 4080)
    assert (
        evaluation.max_workers,
        evaluation.max_total_searches,
        evaluation.max_total_fetches,
        evaluation.max_wall_time_seconds,
    ) == (8, 200, 336, 4080)

    resolved = config.resolve_research_options(
        ResearchOptions(max_workers=5),
        "standard",
        "drb_evaluation",
    )
    assert resolved == {
        "max_rounds": 4,
        "max_worker_iterations": 4,
        "max_workers": 5,
    }

    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    )
    request = service._resolve_request_mode(
        ResearchRequest(
            question="Evaluate this research task",
            mode=ResearchMode.STANDARD,
            profile_id="drb_evaluation",
        )
    )
    budget = service._default_budget(request.mode, request.profile_id)
    assert request.options.max_workers == 8
    assert request.options.max_rounds == 4
    assert request.options.max_worker_iterations == 4
    assert (
        budget.max_total_searches,
        budget.max_total_fetches,
        budget.max_wall_time_seconds,
    ) == (200, 336, 4080)


def test_evaluation_writer_and_reviewer_have_independent_time_slots(monkeypatch):
    config = load_runtime_config()
    synthesizer = Synthesizer(FakeLLM(), runtime_config=config)
    state = {
        "started_at": 1_000.0,
        "request": {"mode": "standard", "profile_id": "drb_evaluation"},
        "run_budget": {"max_wall_time_seconds": 4080},
    }

    monkeypatch.setattr("deepresearch.agents.time.time", lambda: 1_000.0 + 3600)
    assert synthesizer._writer_timeout_seconds(state) == 1220

    monkeypatch.setattr("deepresearch.agents.time.time", lambda: 1_000.0 + 4200)
    assert synthesizer._reviewer_timeout_seconds(state) == 1020

    monkeypatch.setattr("deepresearch.agents.time.time", lambda: 1_000.0 + 5300)
    assert synthesizer._rewrite_timeout_seconds(state) == 1220

    monkeypatch.setattr("deepresearch.agents.time.time", lambda: 1_000.0 + 7000)
    assert synthesizer._reviewer_timeout_seconds(state, recheck=True) == 500

    monkeypatch.setattr("deepresearch.agents.time.time", lambda: 1_000.0 + 8100)
    assert synthesizer._reviewer_timeout_seconds(state, recheck=True) == 0


def test_model_router_switches_provider_and_model_by_mode(monkeypatch):
    raw = _config().model_dump(mode="json")
    raw["llm_providers"] = {
        "fast": {
            "api_key_env": "FAST_KEY",
            "base_url_env": "FAST_URL",
            "base_url": "https://fast.invalid/v1",
        },
        "strong": {
            "api_key_env": "STRONG_KEY",
            "base_url_env": "STRONG_URL",
            "base_url": "https://strong.invalid/v1",
        },
    }
    raw["model_tiers"] = {
        "lite": {"provider": "fast", "model": "fast-lite"},
        "default": {"provider": "fast", "model": "fast-default"},
        "craft": {"provider": "strong", "model": "strong-craft"},
    }
    config = RuntimeConfig.model_validate(raw)
    monkeypatch.setenv("FAST_KEY", "test-fast")
    monkeypatch.setenv("STRONG_KEY", "test-strong")

    router = ModelRouter(config)
    assert router.for_stage("planner").model_id == "fast-default"
    assert router.for_stage("planner").provider_id == "fast"
    token = router.activate("deep")
    try:
        assert router.for_stage("planner").model_id == "strong-craft"
        assert router.for_stage("planner").provider_id == "strong"
    finally:
        router.reset(token)


@pytest.mark.asyncio
async def test_final_stage_router_retries_then_fails_over(monkeypatch):
    raw = _config().model_dump(mode="json")
    raw["llm_recovery"] = {
        "synthesizer": {
            "max_attempts_per_model": 2,
            "backoff_initial_seconds": 0,
            "fallback_tiers": ["default"],
        },
        "worker_agent_loop": {"max_attempts_per_model": 1},
        "report_writer": {"max_attempts_per_model": 1},
        "judge": {"max_attempts_per_model": 1},
    }
    config = RuntimeConfig.model_validate(raw)
    calls: list[str] = []

    class RoutedFake:
        def __init__(self, name: str, *, fails: bool) -> None:
            self.model_id = name
            self.provider_id = name
            self.fails = fails

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.model_id)
            if self.fails:
                raise RuntimeError(f"{self.model_id} unavailable")
            return {"decision": "ready"}, LLMResponse("{}")

    router = ModelRouter(config)
    clients = {
        "craft": RoutedFake("craft-primary", fails=True),
        "default": RoutedFake("default-fallback", fails=False),
    }
    monkeypatch.setattr(router, "resolve_tier", lambda tier: clients[tier])
    value, _ = await router.for_stage("synthesizer").complete_json(
        "system", "{}", max_tokens=128
    )

    assert value == {"decision": "ready"}
    assert calls == ["craft-primary", "craft-primary", "default-fallback"]


@pytest.mark.asyncio
async def test_worker_agent_loop_fails_over_after_transient_model_error(monkeypatch):
    raw = _config().model_dump(mode="json")
    raw["llm_recovery"]["worker_agent_loop"] = {
        "max_attempts_per_model": 1,
        "backoff_initial_seconds": 0,
        "attempt_timeout_seconds": 240,
        "fallback_tiers": ["craft"],
    }
    config = RuntimeConfig.model_validate(raw)
    calls: list[str] = []

    class RoutedFake:
        def __init__(self, name: str, *, fails: bool) -> None:
            self.model_id = name
            self.provider_id = name
            self.fails = fails

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.model_id)
            if self.fails:
                raise TimeoutError(f"{self.model_id} timed out")
            return {"selected_candidate_ids": ["candidate-1"]}, LLMResponse("{}")

    router = ModelRouter(config)
    clients = {
        "default": RoutedFake("worker-primary", fails=True),
        "craft": RoutedFake("worker-fallback", fails=False),
    }
    monkeypatch.setattr(router, "resolve_tier", lambda tier: clients[tier])

    value, _ = await router.for_stage("worker_agent_loop").complete_json(
        "system", "{}", max_tokens=128
    )

    assert value == {"selected_candidate_ids": ["candidate-1"]}
    assert calls == ["worker-primary", "worker-fallback"]


def _hedged_router_config() -> RuntimeConfig:
    raw = _config().model_dump(mode="json")
    raw["research_modes"]["standard"]["stages"]["planner"] = "craft"
    raw["llm_recovery"]["planner"] = {
        "max_attempts_per_model": 1,
        "backoff_initial_seconds": 0,
        "attempt_timeout_seconds": 0.30,
        "recovery_strategy": "hedged",
        "fast_failure_window_seconds": 0.05,
        "fast_failure_retries": 1,
        "hedge_after_seconds": 0.08,
        "fallback_attempt_timeout_seconds": 0.08,
        "primary_result_grace_seconds": 0.02,
        "fallback_tiers": ["default"],
    }
    return RuntimeConfig.model_validate(raw)


@pytest.mark.asyncio
async def test_hedged_router_keeps_primary_and_prefers_it_when_it_finishes(monkeypatch):
    config = _hedged_router_config()
    calls: list[str] = []
    fallback_cancelled = asyncio.Event()

    class RoutedFake:
        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = tier
            self.provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.tier)
            if self.tier == "craft":
                await asyncio.sleep(0.14)
                return {"tasks": [{"objective": "primary"}]}, LLMResponse("{}")
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                fallback_cancelled.set()
                raise
            return {"tasks": [{"objective": "fallback"}]}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: RoutedFake(tier))
    value, _ = await router.for_stage("planner").complete_json(
        "system", "{}", max_tokens=128
    )

    assert value["tasks"][0]["objective"] == "primary"
    assert calls == ["craft", "default"]
    assert fallback_cancelled.is_set()


@pytest.mark.asyncio
async def test_hedged_router_accepts_fallback_and_cancels_slow_primary(monkeypatch):
    config = _hedged_router_config()
    primary_cancelled = asyncio.Event()

    class RoutedFake:
        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = tier
            self.provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            if self.tier == "craft":
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    primary_cancelled.set()
                    raise
            return {"tasks": [{"objective": self.tier}]}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: RoutedFake(tier))
    value, _ = await router.for_stage("planner").complete_json(
        "system", "{}", max_tokens=128
    )

    assert value["tasks"][0]["objective"] == "default"
    assert primary_cancelled.is_set()


@pytest.mark.asyncio
async def test_hedged_router_retries_fast_transient_primary_before_fallback(monkeypatch):
    config = _hedged_router_config()
    calls: list[str] = []

    class RoutedFake:
        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = tier
            self.provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.tier)
            if self.tier == "craft" and calls.count("craft") == 1:
                raise ConnectionError("temporary connection failure")
            return {"tasks": [{"objective": "retry"}]}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: RoutedFake(tier))
    value, _ = await router.for_stage("planner").complete_json(
        "system", "{}", max_tokens=128
    )

    assert value["tasks"][0]["objective"] == "retry"
    assert calls == ["craft", "craft"]


@pytest.mark.asyncio
async def test_hedged_router_cleans_up_both_requests_when_parent_is_cancelled(monkeypatch):
    config = _hedged_router_config()
    cancelled: set[str] = set()
    fallback_started = asyncio.Event()

    class RoutedFake:
        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = tier
            self.provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            if self.tier == "default":
                fallback_started.set()
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.add(self.tier)
                raise
            return {"tasks": [{"objective": self.tier}]}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: RoutedFake(tier))
    call = asyncio.create_task(
        router.for_stage("planner").complete_json(
            "system", "{}", max_tokens=128
        )
    )
    await asyncio.wait_for(fallback_started.wait(), timeout=0.25)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert cancelled == {"craft", "default"}


@pytest.mark.asyncio
async def test_hedged_policy_does_not_apply_to_non_craft_quick_tier(monkeypatch):
    config = _hedged_router_config()
    calls: list[str] = []

    class RoutedFake:
        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = tier
            self.provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.tier)
            return {"tasks": []}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: RoutedFake(tier))
    token = router.activate("quick")
    try:
        await router.for_stage("planner").complete_json(
            "system", "{}", max_tokens=128
        )
    finally:
        router.reset(token)

    assert calls == ["lite"]


@pytest.mark.asyncio
async def test_injected_llm_is_not_retried_by_final_stage_router():
    config = _config()

    class InjectedFailure:
        model_id = "injected"

        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            self.calls += 1
            raise RuntimeError("one injected failure")

    injected = InjectedFailure()
    router = ModelRouter(config, injected_llm=injected)
    with pytest.raises(RuntimeError, match="one injected failure"):
        await router.for_stage("report_writer").complete_json(
            "system", "{}", max_tokens=128
        )
    assert injected.calls == 1


def test_runtime_config_validates_mcp_transport_and_allowlist():
    config = RuntimeConfig.model_validate(
        {
            "llm_output_tokens": _config().llm_output_tokens.model_dump(),
            "mcp": {
                "enabled": True,
                "servers": {
                    "github": {
                        "transport": "streamable_http",
                        "url": "https://example.com/mcp",
                        "tools": {
                            "search_code": {
                                "skill_id": "github-code-search",
                                "result_kind": "search_hits",
                            }
                        },
                    }
                },
            },
        }
    )

    assert config.mcp.servers["github"].tools["search_code"].result_kind == "search_hits"
    with pytest.raises(ValueError, match="require url"):
        RuntimeConfig.model_validate(
            {
                "llm_output_tokens": _config().llm_output_tokens.model_dump(),
                "mcp": {"enabled": True, "servers": {"bad": {"transport": "streamable_http"}}},
            }
        )


def test_runtime_config_can_be_loaded_from_an_explicit_yaml(tmp_path: Path):
    path = tmp_path / "custom-runtime.yaml"
    path.write_text(
        """\
llm_output_tokens:
  planner: 3456
  worker_strategy: 1000
  worker_reflection: 900
  worker_agent_loop: 1000
  evidence_extraction_min: 256
  evidence_extraction_max: 1200
  synthesizer_assessment: 4567
  report_writer: 3000
  extraction_context: 24000
""",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.llm_output_tokens.planner == 3456
    assert config.llm_output_tokens.synthesizer_assessment == 4567


def test_runtime_config_rejects_inverted_extraction_limits():
    with pytest.raises(ValueError, match="evidence_extraction_min"):
        _config(evidence_extraction_min=1300, evidence_extraction_max=1200)


def test_skill_can_be_explicitly_disabled_even_when_its_key_is_configured(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DEEPRESEARCH_DISABLE_TAVILY", "1")

    skills = {item.skill_id: item for item in discover_skills(Path(__file__).resolve().parent.parent)}

    assert skills["tavily-search"].enabled is False


def test_false_like_disable_env_does_not_disable_skill(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("DEEPRESEARCH_DISABLE_TAVILY", "0")

    skills = {item.skill_id: item for item in discover_skills(Path(__file__).resolve().parent.parent)}

    assert skills["tavily-search"].enabled is True


class _PlannerTokenCaptureLLM:
    model_id = "token-capture"

    def __init__(self) -> None:
        self.max_tokens: int | None = None

    async def complete_json(self, system: str, user: str, *, max_tokens: int):
        self.max_tokens = max_tokens
        return {
            "query_type": "straightforward",
            "research_strategy": "one focused task",
            "answer_requirements": ["facts"],
            "tasks": [
                {
                    "objective": "Find the facts",
                    "research_dimension": "facts",
                    "queries": ["subject facts"],
                    "key_questions": ["What are the facts?"],
                    "preferred_source_types": ["primary"],
                    "verification_strategy": "cross-check",
                    "expected_output": "verified facts",
                }
            ],
        }, LLMResponse("{}")


@pytest.mark.asyncio
async def test_planner_and_synthesizer_receive_injected_runtime_config():
    llm = _PlannerTokenCaptureLLM()
    config = _config(planner=3456, synthesizer_assessment=4567)
    planner = Planner(llm, runtime_config=config)
    synthesizer = Synthesizer(llm, runtime_config=config)
    scope = ResearchScope(
        question="Who is the subject?",
        mode=ResearchMode.RESEARCH,
        profile_id="general_research",
        objective="Answer the question",
        required_dimensions=["facts"],
        current_date="2026-07-20",
    )

    await planner.plan(
        "runtime-config-test",
        scope,
        TaskBudget(max_searches=4, max_fetches=4, max_wall_time_seconds=30),
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={},
    )

    assert llm.max_tokens == 3456
    assert synthesizer.runtime_config.llm_output_tokens.synthesizer_assessment == 4567
