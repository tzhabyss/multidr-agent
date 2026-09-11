from __future__ import annotations

import asyncio

import pytest

from deepresearch.agents import (
    Planner,
    _normalize_writer_draft_payload,
    _validate_planner_payload,
    _validate_review_payload,
)
from deepresearch.llm import LLMJSONError, LLMResponse
from deepresearch.model_routing import ModelRouter
from deepresearch.runtime_config import RuntimeConfig
from deepresearch.schemas import ResearchMode, ResearchScope, TaskBudget, WriterDraft


def _config() -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "llm_output_tokens": {
                "planner": 1200,
                "worker_strategy": 512,
                "worker_reflection": 512,
                "worker_agent_loop": 512,
                "evidence_extraction_min": 256,
                "evidence_extraction_max": 800,
                "synthesizer_assessment": 1000,
                "report_writer": 1000,
                "extraction_context": 24000,
            }
        }
    )


def test_writer_normalizes_nested_heading_paragraphs():
    payload = {
        "sections": [
            {
                "title": "第一部分",
                "paragraphs": [
                    {"text": "引言"},
                    {
                        "title": "1. 定义与历史",
                        "paragraphs": [{"text": "正文", "claim_ids": ["clm-1"]}],
                    },
                ],
            }
        ]
    }

    draft = WriterDraft.model_validate(_normalize_writer_draft_payload(payload))

    assert [section.title for section in draft.sections] == [
        "第一部分",
        "1. 定义与历史",
    ]
    assert [item.text for item in draft.sections[0].paragraphs] == ["引言"]
    assert [item.text for item in draft.sections[1].paragraphs] == ["正文"]
    assert draft.sections[1].paragraphs[-1].claim_ids == ["clm-1"]


def test_embedded_defaults_preserve_full_planner_craft_window():
    policy = _config().llm_recovery.planner

    assert policy.attempt_timeout_seconds == 1200
    assert policy.hedge_after_seconds == 800
    assert policy.fallback_attempt_timeout_seconds == 400
    assert policy.primary_result_grace_seconds == 400
    assert policy.outer_timeout_seconds() == 1210


@pytest.mark.asyncio
async def test_planner_stage_retries_transient_connection_failure(monkeypatch):
    config = _config()
    raw = config.model_dump(mode="json")
    raw["llm_recovery"]["planner"] = {
        "max_attempts_per_model": 2,
        "backoff_initial_seconds": 0,
        "fallback_tiers": [],
    }
    config = RuntimeConfig.model_validate(raw)
    calls = 0

    class FlakyLLM:
        model_id = "planner-test"
        provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary provider failure")
            return {"tasks": [], "query_type": "straightforward"}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: FlakyLLM())

    value, _ = await router.for_stage("planner").complete_json(
        "planner", "{}", max_tokens=128
    )

    assert value["query_type"] == "straightforward"
    assert calls == 2


@pytest.mark.asyncio
async def test_planner_stage_times_out_a_hung_attempt_and_retries(monkeypatch):
    config = _config()
    raw = config.model_dump(mode="json")
    raw["llm_recovery"]["planner"] = {
        "max_attempts_per_model": 2,
        "backoff_initial_seconds": 0,
        "attempt_timeout_seconds": 0.01,
        "fallback_tiers": [],
    }
    config = RuntimeConfig.model_validate(raw)
    calls = 0

    class HungOnceLLM:
        model_id = "planner-timeout-test"
        provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(1)
            return {"tasks": [], "query_type": "straightforward"}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: HungOnceLLM())

    value, _ = await router.for_stage("planner").complete_json(
        "planner", "{}", max_tokens=128
    )

    assert value["query_type"] == "straightforward"
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_kind", ["empty_response", "empty_tasks"])
async def test_hedged_planner_falls_back_when_craft_returns_no_usable_plan(
    monkeypatch, empty_kind
):
    config = _config()
    raw = config.model_dump(mode="json")
    raw["research_modes"]["standard"]["stages"]["planner"] = "craft"
    raw["llm_recovery"]["planner"] = {
        "max_attempts_per_model": 1,
        "backoff_initial_seconds": 0,
        "attempt_timeout_seconds": 0.2,
        "recovery_strategy": "hedged",
        "fast_failure_window_seconds": 0.01,
        "fast_failure_retries": 0,
        "hedge_after_seconds": 0.1,
        "fallback_attempt_timeout_seconds": 0.1,
        "primary_result_grace_seconds": 0,
        "fallback_tiers": ["default"],
    }
    config = RuntimeConfig.model_validate(raw)
    calls: list[str] = []

    class TierLLM:
        provider_id = "test"

        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = f"planner-{tier}"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.tier)
            if self.tier == "craft":
                if empty_kind == "empty_response":
                    raise LLMJSONError(
                        "LLM did not return a JSON object", LLMResponse("")
                    )
                return {"tasks": []}, LLMResponse('{"tasks": []}')
            return {
                "query_type": "straightforward",
                "tasks": [{"objective": "fallback plan"}],
            }, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: TierLLM(tier))

    value, _ = await router.for_stage("planner").complete_json_validated(
        "planner",
        "{}",
        max_tokens=128,
        validator=_validate_planner_payload,
    )

    assert value["tasks"][0]["objective"] == "fallback plan"
    assert calls == ["craft", "default"]


@pytest.mark.asyncio
async def test_report_writer_falls_back_once_without_retrying_primary(monkeypatch):
    config = _config()
    raw = config.model_dump(mode="json")
    raw["llm_recovery"]["report_writer"] = {
        "max_attempts_per_model": 1,
        "backoff_initial_seconds": 0,
        "attempt_timeout_seconds": 500,
        "fallback_tiers": ["default"],
    }
    config = RuntimeConfig.model_validate(raw)
    calls: list[str] = []

    class TierLLM:
        provider_id = "test"

        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = f"writer-{tier}"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.tier)
            if self.tier == "craft":
                raise ConnectionError("primary unavailable")
            return {"sections": []}, LLMResponse("{}")

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: TierLLM(tier))

    value, _ = await router.for_stage("report_writer").complete_json(
        "writer", "{}", max_tokens=128
    )

    assert value == {"sections": []}
    assert calls == ["craft", "default"]


@pytest.mark.asyncio
async def test_reviewer_retries_empty_primary_then_falls_back(monkeypatch):
    config = _config()
    raw = config.model_dump(mode="json")
    raw["research_modes"]["standard"]["stages"]["reviewer"] = "craft"
    raw["llm_recovery"]["reviewer"] = {
        "max_attempts_per_model": 2,
        "backoff_initial_seconds": 0,
        "attempt_timeout_seconds": 1,
        "fallback_tiers": ["default"],
    }
    config = RuntimeConfig.model_validate(raw)
    calls: list[str] = []

    class TierLLM:
        provider_id = "test"

        def __init__(self, tier: str):
            self.tier = tier
            self.model_id = f"reviewer-{tier}"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            calls.append(self.tier)
            if self.tier == "craft":
                raise LLMJSONError(
                    "LLM did not return a JSON object", LLMResponse("")
                )
            return {"decision": "pass", "issues": []}, LLMResponse(
                '{"decision":"pass","issues":[]}'
            )

    router = ModelRouter(config)
    monkeypatch.setattr(router, "resolve_tier", lambda tier: TierLLM(tier))

    value, _ = await router.for_stage("reviewer").complete_json_validated(
        "reviewer",
        "{}",
        max_tokens=128,
        validator=_validate_review_payload,
    )

    assert value["decision"] == "pass"
    assert calls == ["craft", "craft", "default"]


def test_hedged_error_selection_preserves_completed_response_over_timeout():
    response = LLMResponse(
        '{"tasks":',
        input_tokens=100,
        output_tokens=40000,
        finish_reason="length",
    )
    malformed = LLMJSONError("LLM returned malformed JSON", response)

    selected = ModelRouter._most_informative_error(
        [malformed, asyncio.TimeoutError()]
    )

    assert selected is malformed
    assert selected.response.finish_reason == "length"


@pytest.mark.asyncio
async def test_first_planner_failure_preserves_response_without_locking_generic_requirements():
    response = LLMResponse(
        '{"query_type":"breadth_first","requirements":[',
        input_tokens=321,
        output_tokens=40000,
        finish_reason="length",
    )

    class TruncatedPlannerLLM:
        model_id = "planner-truncated"
        provider_id = "test"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            assert max_tokens == 1200
            raise LLMJSONError("LLM returned malformed JSON", response)

    planner = Planner(TruncatedPlannerLLM(), runtime_config=_config())
    scope = ResearchScope(
        question="Build the requested complete comparison table.",
        mode=ResearchMode.STANDARD,
        profile_id="drb_evaluation",
        objective="Cover every requested field and comparison.",
        required_dimensions=[
            "题目要求与事实数据",
            "比较、推导与机制分析",
            "来源、口径与不确定性",
        ],
        current_date="2026-08-12",
    )

    plan, tasks, _, trace = await planner.plan(
        "planner-truncation-regression",
        scope,
        TaskBudget(max_searches=12, max_fetches=12, max_wall_time_seconds=120),
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={"prior_plans": []},
    )

    # Emergency search tasks may still be emitted, but a failed first Planner
    # must not promote the three profile defaults into the durable contract.
    assert tasks
    assert "requirements" not in plan
    assert "delivery_contract" not in plan
    assert trace.raw_text == response.content
    assert trace.finish_reason == "length"
    assert trace.input_tokens == 321
    assert trace.output_tokens == 40000
    assert trace.error == "LLMJSONError: LLM returned malformed JSON"
