from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepresearch.plan_policy import (
    estimate_remaining_waves,
    ready_tasks,
    select_wave,
    task_runtime_seed,
    validate_plan,
)
from deepresearch.graph import ResearchGraph, _planner_execution_capacity
from deepresearch.llm import LLMResponse
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import (
    ResearchMode,
    ResearchRequest,
    ResearchScope,
    ResearchTask,
    RunBudget,
    TaskBudget,
)
from deepresearch.service import ResearchService

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


def _task(task_id: str, *, kind: str = "gather", depends_on: list[str] | None = None):
    return ResearchTask(
        task_id=task_id,
        objective=task_id,
        research_dimension=task_id,
        task_kind=kind,
        depends_on=depends_on or [],
        capability={"gather": "web_research", "analyze": "source_analysis", "verify": "claim_verification"}[kind],
        queries=[task_id],
        budget=TaskBudget(max_searches=1, max_fetches=1, max_wall_time_seconds=20),
    )


def test_plan_policy_rejects_cycle_and_falls_back_to_gather():
    tasks, validation = validate_plan(
        [_task("a", depends_on=["b"]), _task("b", depends_on=["a"])],
        max_depth=4,
    )
    assert validation.fallback_used
    assert "cycle_detected" in validation.errors
    assert all(not task.depends_on and task.task_kind == "gather" for task in tasks)


def test_plan_policy_requires_analyze_anchor_and_keeps_stable_ready_order():
    tasks, validation = validate_plan(
        [_task("a"), _task("c", kind="analyze")], max_depth=4
    )
    assert validation.fallback_used
    assert any(error.startswith("unanchored_analyze") for error in validation.errors)
    ready, blocked = ready_tasks(
        [_task("a"), _task("b", depends_on=["a"])],
        [
            {"task_id": "a", "status": "completed"},
            {"task_id": "b", "status": "pending"},
        ],
    )
    assert [task.task_id for task in select_wave(ready, wave_size=1, source_origin_first=True)] == ["b"]
    assert blocked == []


def test_partial_dependency_with_material_is_ready_but_empty_insufficient_is_blocked():
    tasks = [_task("origin"), _task("downstream", depends_on=["origin"])]

    ready, blocked = ready_tasks(
        tasks,
        [
            {"task_id": "origin", "status": "partial", "dependency_usable": True},
            {"task_id": "downstream", "status": "pending"},
        ],
    )
    assert [task.task_id for task in ready] == ["downstream"]
    assert blocked == []

    ready, blocked = ready_tasks(
        tasks,
        [
            {"task_id": "origin", "status": "insufficient", "dependency_usable": False},
            {"task_id": "downstream", "status": "pending"},
        ],
    )
    assert ready == []
    assert blocked[0]["stop_reason"] == "dependency_unusable"


@pytest.mark.asyncio
async def test_controller_stops_only_after_saturated_consecutive_low_gain_waves():
    graph = object.__new__(ResearchGraph)
    graph.planner = SimpleNamespace(runtime_config=load_runtime_config())
    tasks = [
        _task("done-a"),
        _task("done-b"),
        _task("pending"),
    ]
    state = {
        "run_id": "low-gain-run",
        "round_no": 1,
        "wave_no": 4,
        "current_tasks": [task.model_dump(mode="json") for task in tasks],
        "task_runtime": [
            {"task_id": "done-a", "status": "completed"},
            {"task_id": "done-b", "status": "completed"},
            {"task_id": "pending", "status": "pending"},
        ],
        "wave_history": [
            {
                "wave_id": "r1-w1",
                "round_no": 1,
                "wave_no": 1,
                "source_delta": 4,
                "evidence_delta": 8,
                "claim_delta": 8,
                "requirement_support_delta": 4,
            },
            {
                "wave_id": "r1-w2",
                "round_no": 1,
                "wave_no": 2,
                "source_delta": 3,
                "evidence_delta": 6,
                "claim_delta": 6,
                "requirement_support_delta": 1,
            },
            {
                "wave_id": "r1-w3",
                "round_no": 1,
                "wave_no": 3,
                "source_delta": 3,
                "evidence_delta": 5,
                "claim_delta": 5,
                "requirement_support_delta": 1,
            },
            {
                "wave_id": "r1-w4",
                "round_no": 1,
                "wave_no": 4,
                "source_delta": 2,
                "evidence_delta": 4,
                "claim_delta": 4,
                "requirement_support_delta": 1,
            },
        ],
        "requirement_status": [
            {
                "requirement_id": f"req-{index}",
                "importance": "required",
                "status": "partial" if index <= 3 else "missing",
            }
            for index in range(1, 5)
        ],
    }

    result = await graph._control_wave(state)

    assert result["controller_decision"]["route"] == "synthesize_research"
    assert result["controller_decision"]["reason"] == "low_gain_limit"
    assert result["controller_decision"]["required_coverage_ratio"] == 0.75
    assert result["controller_decision"]["low_gain_signal_count"] == 4


@pytest.mark.asyncio
async def test_controller_stops_when_three_signals_match_despite_high_value_task():
    graph = object.__new__(ResearchGraph)
    graph.planner = SimpleNamespace(runtime_config=load_runtime_config())
    protected = _task("protected")
    protected = protected.model_copy(
        update={
            "expected_value": 0.9,
            "produces_requirement_ids": ["req-4"],
        }
    )
    state = {
        "run_id": "protected-low-gain-run",
        "round_no": 1,
        "wave_no": 4,
        "current_tasks": [protected.model_dump(mode="json")],
        "task_runtime": [{"task_id": "protected", "status": "pending"}],
        "wave_history": [
            {
                "wave_id": f"r1-w{index}",
                "round_no": 1,
                "wave_no": index,
                "source_delta": 2,
                "evidence_delta": 2,
                "claim_delta": 2,
                "requirement_support_delta": 1,
            }
            for index in range(1, 5)
        ],
        "requirement_status": [
            {
                "requirement_id": f"req-{index}",
                "importance": "required",
                "status": "partial" if index <= 3 else "missing",
            }
            for index in range(1, 5)
        ],
    }

    result = await graph._control_wave(state)

    assert result["controller_decision"]["route"] == "synthesize_research"
    assert result["controller_decision"]["reason"] == "low_gain_limit"
    assert result["controller_decision"]["low_gain_signal_count"] == 3
    assert result["controller_decision"]["protected_high_value_task_ids"] == [
        "protected"
    ]


@pytest.mark.asyncio
async def test_controller_requires_four_waves_even_when_other_signals_match():
    graph = object.__new__(ResearchGraph)
    graph.planner = SimpleNamespace(runtime_config=load_runtime_config())
    pending = _task("pending")
    state = {
        "run_id": "minimum-wave-hard-gate",
        "round_no": 1,
        "wave_no": 3,
        "current_tasks": [pending.model_dump(mode="json")],
        "task_runtime": [{"task_id": "pending", "status": "pending"}],
        "wave_history": [
            {
                "wave_id": f"r1-w{index}",
                "round_no": 1,
                "wave_no": index,
                "source_delta": 2,
                "evidence_delta": 2,
                "claim_delta": 2,
                "requirement_support_delta": 0,
            }
            for index in range(1, 4)
        ],
        "requirement_status": [
            {
                "requirement_id": f"req-{index}",
                "importance": "required",
                "status": "partial",
            }
            for index in range(1, 5)
        ],
    }

    result = await graph._control_wave(state)

    assert result["controller_decision"]["route"] == "schedule_wave"
    assert result["controller_decision"]["low_gain_signal_count"] == 3
    assert result["controller_decision"]["low_gain_stop_signals"] == {
        "minimum_waves_reached": False,
        "coverage_saturated": True,
        "consecutive_low_gain": True,
        "no_high_value_missing_task": True,
    }


@pytest.mark.asyncio
async def test_controller_continues_when_only_two_low_gain_signals_match():
    graph = object.__new__(ResearchGraph)
    graph.planner = SimpleNamespace(runtime_config=load_runtime_config())
    pending = _task("pending")
    state = {
        "run_id": "two-signal-run",
        "round_no": 1,
        "wave_no": 2,
        "current_tasks": [pending.model_dump(mode="json")],
        "task_runtime": [{"task_id": "pending", "status": "pending"}],
        "wave_history": [
            {
                "wave_id": f"r1-w{index}",
                "round_no": 1,
                "wave_no": index,
                "source_delta": 2,
                "evidence_delta": 2,
                "claim_delta": 2,
                "requirement_support_delta": 0,
            }
            for index in range(1, 3)
        ],
        "requirement_status": [
            {
                "requirement_id": f"req-{index}",
                "importance": "required",
                "status": "partial" if index <= 2 else "missing",
            }
            for index in range(1, 5)
        ],
    }

    result = await graph._control_wave(state)

    assert result["controller_decision"]["route"] == "schedule_wave"
    assert result["controller_decision"]["low_gain_signal_count"] == 2


@pytest.mark.asyncio
async def test_controller_does_not_saturate_repeated_collection_with_open_cells():
    graph = object.__new__(ResearchGraph)
    graph.planner = SimpleNamespace(runtime_config=load_runtime_config())
    completion = _task("complete-row")
    completion = completion.model_copy(
        update={"expected_value": 0.9, "produces_requirement_ids": ["cell-2"]}
    )
    state = {
        "run_id": "repeated-coverage-run",
        "round_no": 1,
        "wave_no": 4,
        "current_tasks": [completion.model_dump(mode="json")],
        "task_runtime": [{"task_id": "complete-row", "status": "pending"}],
        "wave_history": [
            {
                "wave_id": f"r1-w{index}",
                "round_no": 1,
                "wave_no": index,
                "source_delta": 2,
                "evidence_delta": 2,
                "claim_delta": 2,
                "requirement_support_delta": 1,
            }
            for index in range(1, 5)
        ],
        "requirement_status": [
            {"requirement_id": "column-country", "importance": "required", "status": "partial"},
            {"requirement_id": "cell-2", "importance": "required", "status": "partial"},
        ],
        "collection_coverage": {
            "active": True,
            "known_case_count": 1,
            "verified_case_count": 0,
            "expected_cell_count": 3,
            "covered_cell_count": 1,
            "case_coverage_ratio": 0.0,
            "cell_coverage_ratio": 0.3333,
            "inventory_open": True,
            "incomplete_cell_requirement_ids": ["cell-2"],
            "actionable_task_ids": ["complete-row"],
        },
    }

    result = await graph._control_wave(state)

    decision = result["controller_decision"]
    assert decision["low_gain_stop_signals"]["coverage_saturated"] is False
    assert decision["route"] == "schedule_wave"
    assert decision["protected_high_value_task_ids"] == ["complete-row"]


def _four_wave_tasks() -> list[ResearchTask]:
    return [
        _task("origin"),
        _task("table", depends_on=["origin"]),
        _task("outcomes", kind="analyze", depends_on=["origin"]),
        _task("mortality", kind="analyze", depends_on=["origin"]),
        _task("age", kind="analyze", depends_on=["table", "outcomes"]),
        _task("duration", kind="analyze", depends_on=["table", "outcomes"]),
        _task("risk", kind="analyze", depends_on=["outcomes"]),
        _task(
            "verify",
            kind="verify",
            depends_on=["table", "outcomes", "mortality", "age", "duration", "risk"],
        ),
    ]


def test_remaining_wave_estimate_preserves_four_level_dag_capacity():
    tasks = _four_wave_tasks()
    assert estimate_remaining_waves(
        tasks,
        task_runtime_seed(tasks),
        wave_size=3,
        source_origin_first=True,
    ) == 4


@pytest.mark.asyncio
async def test_wave_budget_is_dynamic_and_capped_at_900_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    now = 10_000.0
    monkeypatch.setattr("deepresearch.graph.time.time", lambda: now)
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "dynamic-budget",
    )
    tasks = _four_wave_tasks()
    scope = ResearchScope(
        question="Budget a four-wave DAG",
        mode=ResearchMode.STANDARD,
        profile_id="general_research",
        objective="Budget a four-wave DAG",
        required_dimensions=[],
        current_date="2026-08-06",
    )
    async with service:
        assert service._graph is not None
        scheduled = await service._graph._schedule_wave(
            {
                "run_id": "dynamic-budget-run",
                "request": ResearchRequest(
                    question="Budget a four-wave DAG",
                    mode=ResearchMode.STANDARD,
                ).model_dump(mode="json"),
                "scope": scope.model_dump(mode="json"),
                    "run_budget": RunBudget(
                        max_total_searches=200,
                        max_total_fetches=336,
                            max_wall_time_seconds=4080,
                ).model_dump(mode="json"),
                "started_at": now,
                "round_no": 1,
                "wave_no": 0,
                "current_tasks": [item.model_dump(mode="json") for item in tasks],
                "task_runtime": task_runtime_seed(tasks),
                "requirement_status": [],
                "claims": [],
                "sources": [],
                "evidence": [],
            }
        )

    budget = scheduled["wave_budget"]
    assert budget["estimated_remaining_waves"] == 4
    assert budget["research_finalization_reserve_seconds"] == 180
    assert budget["wave_budget_seconds"] == 900
    assert budget["worker_ceiling_seconds"] == 900
    assert scheduled["current_wave_tasks"][0]["task_id"] == "origin"
    assert scheduled["wave_tool_deadline_at"] < scheduled["wave_deadline_at"]


@pytest.mark.asyncio
async def test_wave_merge_marks_incomplete_material_as_partial_dependency():
    task = _task("origin")
    task = task.model_copy(update={"produces_requirement_ids": ["req-001"]})
    merged = await ResearchGraph._merge_wave(
        ResearchGraph,
        {
            "run_id": "run-partial-dependency",
            "round_no": 1,
            "wave_no": 1,
            "current_wave_tasks": [task.model_dump(mode="json")],
            "contributions": [
                {
                    "task_id": "origin",
                    "status": "failed",
                    "usable": False,
                    "sufficiency_passed": False,
                    "stop_reason": "insufficient_evidence",
                    "source_ids": ["src-1"],
                    "evidence_ids": ["ev-1"],
                    "candidate_claim_ids": ["claim-1"],
                    "candidate_lead_ids": [],
                }
            ],
            "claims": [
                {"claim_id": "claim-1", "requirement_ids": ["req-001"]}
            ],
            "sources": [{"source_id": "src-1"}],
            "evidence": [{"evidence_id": "ev-1"}],
            "wave_baseline": {
                "source_count": 0,
                "evidence_count": 0,
                "claim_count": 0,
                "covered_requirement_ids": [],
                "started_at": 0,
            },
        },
    )

    runtime = merged["task_runtime"][0]
    assert runtime["status"] == "partial"
    assert runtime["dependency_usable"] is True
    assert runtime["sufficiency_passed"] is False


def test_partial_dependency_projects_its_material_to_downstream_worker():
    context = ResearchGraph._dependency_context(
        {
            "task_runtime": [
                {
                    "task_id": "origin",
                    "status": "partial",
                    "dependency_usable": True,
                }
            ],
            "contributions": [
                {
                    "task_id": "origin",
                    "status": "failed",
                    "usable": False,
                    "result_brief": "Useful partial inventory",
                    "source_ids": ["src-1"],
                    "evidence_ids": ["ev-1"],
                    "candidate_claim_ids": ["claim-1"],
                }
            ],
            "claims": [
                {
                    "claim_id": "claim-1",
                    "text": "Verified partial claim",
                    "requirement_ids": ["req-001"],
                }
            ],
            "evidence": [
                {
                    "evidence_id": "ev-1",
                    "source_id": "src-1",
                    "quote": "Verified partial quote",
                    "requirement_ids": ["req-001"],
                }
            ],
            "sources": [
                {
                    "source_id": "src-1",
                    "title": "Partial source",
                    "canonical_uri": "https://example.com/source",
                    "quality_tier": 2,
                }
            ],
            "candidate_leads": [],
        },
        {"task_id": "downstream", "depends_on": ["origin"]},
    )

    assert context["dependencies"] == [
        {
            "task_id": "origin",
            "result_brief": "Useful partial inventory",
            "status": "partial",
        }
    ]
    assert [item["claim_id"] for item in context["claims"]] == ["claim-1"]
    assert [item["evidence_id"] for item in context["evidence"]] == ["ev-1"]
    assert [item["source_id"] for item in context["sources"]] == ["src-1"]


def test_replan_capacity_rejects_more_tasks_than_remaining_time_can_execute():
    config = load_runtime_config().graph_engineering

    task_capacity, executable_waves = _planner_execution_capacity(
        509.0,
        config,
        max_workers=3,
    )

    assert executable_waves == 1
    assert task_capacity == 3


@pytest.mark.asyncio
async def test_complete_dag_runs_dependency_in_waves_and_persists_graph_artifacts(tmp_path: Path):
    class DAGLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            payload = json.loads(user)
            if "research planner" in system:
                return {
                    "query_type": "breadth_first",
                    "tasks": [
                        {"task_key": "a", "objective": "A", "research_dimension": "dimension-a", "queries": ["dimension-a"], "sufficiency": {"min_queries": 1, "min_query_coverage": 1, "min_successful_extractions": 1, "min_effective_sources": 1, "min_independent_domains": 1, "min_evidence": 1}},
                        {"task_key": "b", "objective": "B", "research_dimension": "dimension-b", "queries": ["dimension-b"], "sufficiency": {"min_queries": 1, "min_query_coverage": 1, "min_successful_extractions": 1, "min_effective_sources": 1, "min_independent_domains": 1, "min_evidence": 1}},
                        {"task_key": "c", "objective": "C", "research_dimension": "dimension-c", "queries": ["dimension-c"], "depends_on": ["a", "b"], "task_kind": "verify", "capability": "claim_verification", "sufficiency": {"min_queries": 1, "min_query_coverage": 1, "min_successful_extractions": 1, "min_effective_sources": 1, "min_independent_domains": 1, "min_evidence": 1}},
                    ],
                }, LLMResponse("{}", input_tokens=1, output_tokens=1)
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=DAGLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Run a three-step DAG",
                options={"max_rounds": 1, "max_workers": 2, "max_worker_iterations": 1},
            )
        )
    wave_history = json.loads((Path(result.artifact_dir) / "details" / "graph" / "wave-history.json").read_text(encoding="utf-8"))
    task_runtime = json.loads((Path(result.artifact_dir) / "details" / "graph" / "task-runtime.json").read_text(encoding="utf-8"))
    plan = json.loads((Path(result.artifact_dir) / "details" / "planning" / "research-plan.json").read_text(encoding="utf-8"))
    task_ids = {item["research_dimension"]: item["task_id"] for item in plan["tasks"]}
    assert len(wave_history) >= 2
    assert set(wave_history[0]["task_ids"]) == {task_ids["dimension-a"], task_ids["dimension-b"]}
    c_runtime = next(item for item in task_runtime if item["task_id"] == task_ids["dimension-c"])
    assert c_runtime["wave_no"] >= 2


@pytest.mark.asyncio
async def test_graph_feature_flag_preserves_legacy_dispatch(tmp_path: Path):
    config = load_runtime_config()
    config = config.model_copy(
        update={
            "graph_engineering": config.graph_engineering.model_copy(
                update={"enabled": False, "wave_scheduler_enabled": False}
            )
        }
    )
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "legacy",
        runtime_config=config,
    )
    async with service:
        result = await service.run(ResearchRequest(question="Legacy dispatch"))
    history = json.loads(
        (Path(result.artifact_dir) / "details" / "graph" / "wave-history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history == []
