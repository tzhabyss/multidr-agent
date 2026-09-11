from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from deepresearch.agent_loop import AgentLoop
from deepresearch.agents import Planner
from deepresearch.llm import LLMResponse, LLMToolCall
from deepresearch.prompts import (
    EVALUATION_PLANNER_ADDENDUM,
    PLANNER_SYSTEM_PROMPT,
    WORKER_SYSTEM_PROMPT,
    WORKER_TOOL_LOOP_SYSTEM_PROMPT,
)
from deepresearch.schemas import (
    ResearchMode,
    ResearchOptions,
    ResearchRequest,
    ResearchScope,
    SearchHit,
    TaskBudget,
)
from deepresearch.service import ResearchService
from deepresearch.skill_loader import load_skill_registry
from deepresearch.skill_registry import (
    SkillAuthorizationError,
    SkillDefinition,
    SkillRegistry,
)

from .fakes import FakeLLM, FakeSearchBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_skill_loader_discovers_role_scoped_runtime_skills():
    backend = FakeSearchBackend()
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": backend.search},
    )

    worker_tools = registry.to_openai_tools(
        role="worker",
        profile_id="general_research",
        tags=("search",),
    )
    planner_tools = registry.to_openai_tools(
        role="planner",
        profile_id="general_research",
    )

    assert registry.get("tavily_search") is not None
    assert registry.get("web_search") is not None
    assert "Research Planning" in registry.instructions_for(
        role="planner", profile_id="general_research"
    )
    assert any(item["function"]["name"] == "web_search" for item in worker_tools)
    assert planner_tools == []


@pytest.mark.asyncio
async def test_profile_specific_search_fallback_keeps_evaluation_deepseek_only():
    calls: list[str] = []

    async def deepseek_search():
        calls.append("deepseek")
        return [{"provider": "mcp:deepseek-web-search"}]

    async def tavily_search():
        calls.append("tavily")
        return [{"provider": "tavily"}]

    registry = SkillRegistry(
        profile_skill_fallbacks={
            "drb_evaluation": {"web-search": ["deepseek-web-search"]}
        }
    )
    registry.register(
        SkillDefinition(
            skill_id="web-search",
            description="search",
            instructions="",
            roles=frozenset({"worker"}),
            profiles=frozenset({"general_research", "drb_evaluation"}),
            fallback_skills=("deepseek-web-search", "tavily-search"),
        )
    )
    registry.register(
        SkillDefinition(
            skill_id="deepseek-web-search",
            description="deepseek",
            instructions="",
            roles=frozenset({"worker"}),
            profiles=frozenset({"drb_evaluation"}),
            handler=deepseek_search,
        )
    )
    registry.register(
        SkillDefinition(
            skill_id="tavily-search",
            description="tavily",
            instructions="",
            roles=frozenset({"worker"}),
            profiles=frozenset({"general_research", "drb_evaluation"}),
            handler=tavily_search,
        )
    )

    evaluation_result = await registry.execute(
        role="worker",
        profile_id="drb_evaluation",
        skill_name="web-search",
        arguments={},
    )
    assert evaluation_result[0]["provider"] == "mcp:deepseek-web-search"
    assert calls == ["deepseek"]
    evaluation_tools = registry.to_openai_tools(
        role="worker", profile_id="drb_evaluation"
    )
    assert "tavily_search" not in {
        item["function"]["name"] for item in evaluation_tools
    }
    with pytest.raises(SkillAuthorizationError, match="does not select"):
        await registry.execute(
            role="worker",
            profile_id="drb_evaluation",
            skill_name="tavily-search",
            arguments={},
        )

    calls.clear()
    general_result = await registry.execute(
        role="worker",
        profile_id="general_research",
        skill_name="web-search",
        arguments={},
    )
    assert general_result[0]["provider"] == "tavily"
    assert calls == ["tavily"]


@pytest.mark.asyncio
async def test_profile_fanout_merges_selected_search_providers_by_url():
    calls: list[str] = []

    async def first_search(**_arguments):
        calls.append("first")
        return [
            {"title": "A", "url": "https://a.example", "provider": "first"},
            {"title": "Shared", "url": "https://shared.example", "provider": "first"},
        ]

    async def second_search(**_arguments):
        calls.append("second")
        return [
            {"title": "Shared 2", "url": "https://shared.example", "provider": "second"},
            {"title": "B", "url": "https://b.example", "provider": "second"},
        ]

    registry = SkillRegistry(
        profile_skill_fallbacks={
            "drb_evaluation": {"web-search": ["first-search", "second-search"]}
        },
        profile_skill_provider_modes={
            "drb_evaluation": {"web-search": "fanout"}
        },
    )
    registry.register(
        SkillDefinition(
            skill_id="web-search",
            description="parent",
            instructions="",
            roles=frozenset({"worker"}),
            profiles=frozenset({"drb_evaluation"}),
            fallback_skills=("first-search", "second-search"),
        )
    )
    for skill_id, handler in (
        ("first-search", first_search),
        ("second-search", second_search),
    ):
        registry.register(
            SkillDefinition(
                skill_id=skill_id,
                description=skill_id,
                instructions="",
                roles=frozenset({"worker"}),
                profiles=frozenset({"drb_evaluation"}),
                handler=handler,
            )
        )

    result = await registry.execute(
        role="worker",
        profile_id="drb_evaluation",
        skill_name="web-search",
        arguments={"query": "test"},
    )

    assert calls == ["first", "second"]
    assert registry.provider_call_width(
        profile_id="drb_evaluation", skill_name="web-search"
    ) == 2
    assert [item["url"] for item in result] == [
        "https://a.example",
        "https://shared.example",
        "https://b.example",
    ]


@pytest.mark.asyncio
async def test_agent_loop_observes_skill_result_before_finishing():
    backend = FakeSearchBackend()
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": backend.search},
    )

    class ToolCallingLLM:
        model_id = "tool-calling-fixture"

        def __init__(self):
            self.calls = 0

        async def complete_with_tools(self, *, messages, tools, max_tokens=1200):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    input_tokens=10,
                    output_tokens=5,
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            call_id="call-1",
                            name="web_search",
                            arguments={"query": "EV market official data", "max_results": 5},
                        )
                    ],
                )
            assert messages[-1]["role"] == "tool"
            assert "Verified source" in messages[-1]["content"]
            return LLMResponse(
                content="Search coverage is sufficient.",
                input_tokens=8,
                output_tokens=4,
                finish_reason="stop",
            )

    llm = ToolCallingLLM()
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        role="worker",
        profile_id="general_research",
        max_iterations=4,
        max_tool_calls=3,
    )
    result = await loop.run(
        system_prompt="Use search skills and observe results.",
        user_prompt="Research the EV market.",
        tags=("search",),
        max_tokens=1000,
    )

    assert backend.calls == ["EV market official data"]
    assert llm.calls == 2
    assert len(result.executions) == 1
    assert len(result.traces) == 2
    assert result.stop_reason == "model_finished"


@pytest.mark.asyncio
async def test_agent_loop_compacts_model_observation_but_preserves_execution_result():
    long_snippet = "x" * 2_000

    async def long_search(query: str, max_results: int = 5):
        return [
            {
                "title": f"result-{index}",
                "url": f"https://example.com/{index}",
                "snippet": long_snippet,
                "provider": "fixture",
                "rank": index,
            }
            for index in range(10)
        ]

    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": long_search},
    )

    class ObservationLLM:
        model_id = "observation-fixture"

        def __init__(self):
            self.calls = 0
            self.observation = ""

        async def complete_with_tools(self, *, messages, tools, max_tokens=1200):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            call_id="call-long",
                            name="web_search",
                            arguments={"query": "compact observation", "max_results": 10},
                        )
                    ],
                )
            self.observation = messages[-1]["content"]
            return LLMResponse(content="done", finish_reason="stop")

    llm = ObservationLLM()
    result = await AgentLoop(
        llm=llm,
        registry=registry,
        role="worker",
        profile_id="general_research",
        max_iterations=2,
        max_tool_calls=2,
    ).run(
        system_prompt="Search and observe.",
        user_prompt="Test compact observations.",
        tags=("search",),
        max_tokens=1000,
    )

    assert len(result.executions[0].result) == 10
    assert len(result.executions[0].result[0]["snippet"]) == 2_000
    assert "_omitted_items" in llm.observation
    assert "...[truncated]" in llm.observation
    assert long_snippet not in llm.observation


@pytest.mark.asyncio
async def test_agent_loop_starts_no_new_tools_after_stage_deadline():
    backend = FakeSearchBackend()
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": backend.search},
    )

    class DeadlineLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.tool_calls = 0

        async def complete_with_tools(self, *, messages, tools, max_tokens=1200):
            self.tool_calls += 1
            return LLMResponse(content="done", finish_reason="stop")

    llm = DeadlineLLM()
    result = await AgentLoop(
        llm=llm,
        registry=registry,
        role="worker",
        profile_id="general_research",
        max_iterations=2,
        max_tool_calls=2,
        tool_deadline_at=0.0,
    ).run(
        system_prompt="Search.",
        user_prompt="Deadline test.",
        tags=("search",),
        max_tokens=1000,
    )

    assert result.stop_reason == "stage_deadline_reached"
    assert result.executions == []
    assert llm.tool_calls == 0
    assert backend.calls == []


@pytest.mark.asyncio
async def test_agent_loop_cancels_inflight_model_call_at_stage_deadline():
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": FakeSearchBackend().search},
    )

    class SlowDeadlineLLM(FakeLLM):
        async def complete_with_tools(self, *, messages, tools, max_tokens=1200):
            await asyncio.sleep(1)
            return LLMResponse(content="late", finish_reason="stop")

    started = time.monotonic()
    result = await AgentLoop(
        llm=SlowDeadlineLLM(),
        registry=registry,
        role="worker",
        profile_id="general_research",
        max_iterations=2,
        max_tool_calls=2,
        tool_deadline_at=time.time() + 0.02,
    ).run(
        system_prompt="Search.",
        user_prompt="Deadline test.",
        tags=("search",),
        max_tokens=1000,
    )

    assert result.stop_reason == "stage_deadline_reached"
    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_planner_can_allocate_more_than_three_workers():
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": FakeSearchBackend().search},
    )
    scope = ResearchScope(
        question="Analyze a broad market",
        mode=ResearchMode.RESEARCH,
        profile_id="general_research",
        objective="Analyze a broad market",
        required_dimensions=[f"dimension-{index}" for index in range(5)],
        current_date="2026-07-20",
        options=ResearchOptions(max_workers=5),
    )

    llm = FakeLLM()
    _, tasks, _, _ = await Planner(llm, registry).plan(
        "scalable-planner",
        scope,
        TaskBudget(max_searches=15, max_fetches=20, max_wall_time_seconds=60),
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={},
    )

    assert len(tasks) == 5
    assert len({task.task_id for task in tasks}) == 5
    planner_payload = json.loads(
        next(call[2] for call in llm.calls if call[0] == "json")
    )
    assert planner_payload["rules"]["target_parallel_tasks_first_wave"] == 5
    assert planner_payload["rules"]["max_logical_searches_per_worker_pass"] == 2
    assert planner_payload["rules"]["max_search_passes_per_task"] == 4
    assert planner_payload["rules"]["max_logical_searches_per_task"] == 8


@pytest.mark.asyncio
async def test_evaluation_planner_adds_generic_source_origin_discovery():
    llm = FakeLLM()
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": FakeSearchBackend().search},
    )
    scope = ResearchScope(
        question=(
            "Conduct a systematic review of online learning and summarize at least "
            "15 studies in a table."
        ),
        mode=ResearchMode.STANDARD,
        profile_id="drb_evaluation",
        objective="Review online learning studies",
        required_dimensions=[],
        suggested_dimensions=["study evidence", "method analysis"],
        current_date="2026-07-24",
        options=ResearchOptions(max_workers=4),
    )

    _, tasks, _, _ = await Planner(llm, registry).plan(
        "source-origin-planner",
        scope,
        TaskBudget(max_searches=16, max_fetches=24, max_wall_time_seconds=60),
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={},
    )

    assert tasks[0].source_origin is True
    assert all(not task.source_origin for task in tasks[1:])
    planner_system = next(call[1] for call in llm.calls if call[0] == "json")
    assert "query is search-engine" in planner_system
    assert "source_origin=true" in planner_system
    assert "candidate inventory" in planner_system
    assert "bibliography" in planner_system
    assert "candidate inventory" in WORKER_SYSTEM_PROMPT
    assert "exact-title" in WORKER_SYSTEM_PROMPT


def test_query_rewrite_few_shot_belongs_to_worker_and_planner_has_complete_dag_example():
    assert "Query rewrite few-shot" in WORKER_SYSTEM_PROMPT
    assert "Treat an explicit date boundary as a hard research constraint" in WORKER_SYSTEM_PROMPT
    assert "Complex user questions require a query portfolio" in WORKER_SYSTEM_PROMPT
    assert "identify the branch failure before rewriting" in WORKER_SYSTEM_PROMPT
    assert "select a complementary evidence set" in WORKER_SYSTEM_PROMPT
    assert "delegated key questions, missing fields, and assigned requirements" in WORKER_SYSTEM_PROMPT
    assert "Apply source authority and timeliness only after a candidate is relevant" in WORKER_SYSTEM_PROMPT
    assert "native function-call interface" in WORKER_TOOL_LOOP_SYSTEM_PROMPT
    assert '"action": "search"' in WORKER_TOOL_LOOP_SYSTEM_PROMPT
    assert "Return exactly one JSON object" not in WORKER_TOOL_LOOP_SYSTEM_PROMPT
    assert "max_logical_searches_per_task" in PLANNER_SYSTEM_PROMPT
    assert "Planner-generated candidates" in PLANNER_SYSTEM_PROMPT
    assert "one independently trackable requirement per case-field cell" in PLANNER_SYSTEM_PROMPT
    assert "Search-query few-shot" not in EVALUATION_PLANNER_ADDENDUM
    assert "Complete DAG few-shot" in EVALUATION_PLANNER_ADDENDUM
    assert '"depends_on": ["origin_inventory"]' in EVALUATION_PLANNER_ADDENDUM
    assert '"task_kind": "analyze"' in EVALUATION_PLANNER_ADDENDUM
    assert "General rubric-grade planning pattern" in EVALUATION_PLANNER_ADDENDUM
    assert "candidate_cases inventory" in EVALUATION_PLANNER_ADDENDUM
    assert "Candidate cases are untrusted leads" in EVALUATION_PLANNER_ADDENDUM
    assert "Intervention Q" in EVALUATION_PLANNER_ADDENDUM
    assert "fictional Intervention Q" in EVALUATION_PLANNER_ADDENDUM
    example = EVALUATION_PLANNER_ADDENDUM.split(
        "A strong Round-1 JSON plan has this shape:\n", 1
    )[1].split("\n\nLearn the invariant relationships:", 1)[0]
    parsed = json.loads(example)
    assert len(parsed["requirements"]) == 10
    task_keys = {task["task_key"] for task in parsed["tasks"]}
    assert task_keys == {
        "origin_inventory",
        "primary_characteristics",
        "overall_outcome",
        "moderator_a",
        "moderator_b",
        "cross_study_synthesis",
    }


@pytest.mark.asyncio
async def test_scope_preserves_more_than_three_explicit_dimensions(tmp_path):
    dimensions = [f"dimension-{index}" for index in range(6)]
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        artifact_root=tmp_path / "artifacts",
    )
    request = ResearchRequest(
        question="Analyze every explicit dimension",
        constraints={"required_dimensions": dimensions},
        options={"max_workers": 6},
    )

    async with service:
        assert service._graph is not None
        normalized = await service._graph._normalize_scope(
            {
                "run_id": "scope-dimensions-test",
                "request": request.model_dump(mode="json"),
            }
        )

    assert normalized["scope"]["required_dimensions"] == dimensions


@pytest.mark.asyncio
async def test_planner_repairs_duplicate_explicit_dimensions():
    class DuplicateDimensionLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            value, response = await super().complete_json(
                system, user, max_tokens=max_tokens
            )
            if "research planner" in system and len(value.get("tasks", [])) >= 4:
                value["tasks"][1]["research_dimension"] = value["tasks"][0][
                    "research_dimension"
                ]
            return value, response

    dimensions = [f"dimension-{index}" for index in range(4)]
    registry = load_skill_registry(
        PROJECT_ROOT,
        runtime_overrides={"web-search": FakeSearchBackend().search},
    )
    scope = ResearchScope(
        question="Analyze all required dimensions",
        mode=ResearchMode.RESEARCH,
        profile_id="general_research",
        objective="Analyze all required dimensions",
        required_dimensions=dimensions,
        current_date="2026-07-20",
        options=ResearchOptions(max_workers=4),
    )

    _, tasks, _, _ = await Planner(DuplicateDimensionLLM(), registry).plan(
        "duplicate-dimension-planner",
        scope,
        TaskBudget(max_searches=12, max_fetches=16, max_wall_time_seconds=60),
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={},
    )

    assert [task.research_dimension for task in tasks] == dimensions
