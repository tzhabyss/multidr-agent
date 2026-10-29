from __future__ import annotations

from pathlib import Path
import hashlib
import json
import asyncio

import pytest

from deepresearch.agents import Planner, ResearchWorker
from deepresearch.schemas import (
    FetchedDocument,
    ResearchMode,
    ResearchRequest,
    ResearchScope,
    RunStatus,
    SearchHit,
    TaskBudget,
)
from deepresearch.llm import LLMResponse, LLMToolCall
from deepresearch.service import ResearchService, RunAlreadyExistsError
from deepresearch.skill_registry import SkillRegistry

from .fakes import (
    FailOneFetcher,
    FakeFetcher,
    FakeLLM,
    FakeSearchBackend,
    TwoHitSearchBackend,
)


class LowControlSearchBackend:
    name = "low_control_search"

    async def search(self, query: str, max_results: int = 5):
        return [
            SearchHit(
                title="User-published post",
                url="https://medium.com/example/unverified-post",
                snippet="The verified fact is 42 units.",
                provider=self.name,
                rank=1,
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_id", "expected_fetches"),
    [("drb_evaluation", 1), ("general_research", 1)],
)
async def test_coarse_candidates_are_visible_regardless_of_quality_hint(
    tmp_path: Path,
    profile_id: str,
    expected_fetches: int,
):
    fetcher = FakeFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=LowControlSearchBackend(),
        fetcher=fetcher,
        artifact_root=tmp_path / profile_id,
    )

    async with service:
        await service.run(
            ResearchRequest(
                question="Verify a public claim",
                profile_id=profile_id,
                options={"max_rounds": 1, "max_worker_iterations": 1},
            )
        )

    assert len(fetcher.calls) == expected_fetches


@pytest.mark.asyncio
async def test_replanner_bundles_multiple_gaps_for_one_research_object():
    class DuplicateObjectPlannerLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "research planner" not in system:
                return await super().complete_json(system, user, max_tokens=max_tokens)
            tasks = [
                {
                    "objective": f"Resolve {dimension} for cxy0714",
                    "research_dimension": dimension,
                    "research_object": "cxy0714",
                    "queries": [f"cxy0714 {dimension}"],
                    "key_questions": [f"What establishes {dimension}?"],
                }
                for dimension in ("identity", "match statistics", "direct comparison")
            ]
            return tasks and {"tasks": tasks}, LLMResponse("{}", input_tokens=20, output_tokens=10)

    planner = Planner(DuplicateObjectPlannerLLM())
    scope = ResearchScope(
        question="Compare cxy0714 and ZmjjKK",
        mode=ResearchMode.STANDARD,
        profile_id="general_research",
        objective="Resolve the comparison",
        required_dimensions=[],
        current_date="2026-07-26",
        options={"max_rounds": 2, "max_workers": 3},
    )
    _, tasks, _, _ = await planner.plan(
        "replan-object-run",
        scope,
        TaskBudget(max_searches=9, max_fetches=9, max_wall_time_seconds=120),
        round_no=2,
        gaps=[
            {"dimension": "identity", "missing_information": "identity"},
            {"dimension": "match statistics", "missing_information": "statistics"},
            {"dimension": "direct comparison", "missing_information": "comparison"},
        ],
        attempted_queries=[],
        evidence_summary={"prior_plans": []},
    )

    assert len(tasks) == 1
    assert tasks[0].research_object == "cxy0714"
    assert len(tasks[0].queries) == len(set(tasks[0].queries))
    assert len(tasks[0].key_questions) == 3


@pytest.mark.asyncio
async def test_worker_react_turn_observes_extracted_claims_before_next_search(
    tmp_path: Path,
):
    class ReactLLM(FakeLLM):
        supports_tool_calling = True

        def __init__(self):
            super().__init__()
            self.react_payloads: list[dict] = []

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "research planner" in system:
                return (
                    {"tasks": [{
                        "objective": "Verify the fact twice",
                        "research_dimension": "fact",
                        "research_object": "fixture fact",
                        "queries": ["planner fallback"],
                        "sufficiency": {
                            "min_queries": 2,
                            "min_query_coverage": 2,
                            "min_successful_extractions": 2,
                            "min_effective_sources": 2,
                            "min_independent_domains": 2,
                            "min_evidence": 2,
                        },
                    }]},
                    LLMResponse("{}", input_tokens=20, output_tokens=10),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

        async def complete_with_tools(self, *, messages, tools, max_tokens):
            if messages[-1]["role"] == "tool":
                return LLMResponse("observed search result", input_tokens=10, output_tokens=5)
            payload = json.loads(messages[1]["content"])
            self.react_payloads.append(payload)
            query = (
                "second evidence query"
                if payload.get("validated_claims")
                else "first evidence query"
            )
            return LLMResponse(
                "",
                input_tokens=10,
                output_tokens=5,
                tool_calls=[
                    LLMToolCall(
                        call_id=f"call-{len(self.react_payloads)}",
                        name="web_search",
                        arguments={"query": query, "max_results": 5},
                    )
                ],
            )

    class ReactSearch:
        name = "react-search"

        def __init__(self):
            self.calls: list[str] = []

        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            host = "first.example" if query.startswith("first") else "second.example"
            return [
                SearchHit(
                    title=query,
                    url=f"https://{host}/source",
                    snippet="The verified fact is 42 units.",
                    provider=self.name,
                    rank=1,
                )
            ]

    llm = ReactLLM()
    search = ReactSearch()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        await service.run(
            ResearchRequest(
                question="Verify one fact from two independent sources",
                options={"max_rounds": 1, "max_worker_iterations": 3},
            )
        )

    assert search.calls == ["first evidence query", "second evidence query"]
    assert len(llm.react_payloads) == 2
    assert llm.react_payloads[0]["validated_claims"] == []
    assert llm.react_payloads[1]["validated_claims"]


@pytest.mark.asyncio
async def test_deep_search_produces_verified_answer_and_artifacts(tmp_path: Path):
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(question="What is the verified fact?", mode=ResearchMode.DEEP_SEARCH)
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert result.gate.passed
    assert "42 units" in result.markdown
    assert len(result.sources) == 1
    assert Path(result.output_path).exists()
    assert (Path(result.artifact_dir) / "details" / "manifest.json").exists()
    assert result.usage["searches"] == 1


@pytest.mark.asyncio
async def test_deep_report_uses_parallel_research_tasks(tmp_path: Path):
    search = FakeSearchBackend()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    request = ResearchRequest(
        question="Prepare a broad report",
        mode=ResearchMode.DEEP_REPORT,
        constraints={
            "required_dimensions": ["facts", "risks", "alternatives"],
        },
    )
    async with service:
        result = await service.run(request)

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert len(search.calls) == 3
    assert "核心发现" in result.markdown
    assert "来源质量说明" in result.markdown
    assert result.usage["searches"] == 3
    assert len(result.sources) == 1
    writer_trace = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "report.json").read_text(encoding="utf-8")
    )
    assert not writer_trace["fallback_used"]
    assert writer_trace["raw_output"]["sections"]


@pytest.mark.asyncio
async def test_deep_report_fetches_multiple_sites_before_stopping(tmp_path: Path):
    class ThreeSiteSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            return [
                {
                    "title": f"Source {index}",
                    "url": f"https://source-{index}.example/research",
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": index,
                }
                for index in range(1, 4)
            ]

    search = ThreeSiteSearch()
    fetcher = FakeFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=search,
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Research one dimension deeply",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
                options={
                    "task_sufficiency": {
                        "min_successful_extractions": 3,
                        "min_effective_sources": 3,
                        "min_independent_domains": 3,
                        "min_evidence": 3,
                    }
                },
            )
        )

    assert len(search.calls) == 1
    assert len(fetcher.calls) == 3
    assert len({source.canonical_uri for source in result.sources}) == 3
    assert result.status == RunStatus.COMPLETED_VERIFIED


@pytest.mark.asyncio
async def test_failed_first_attempt_is_retried_with_feedback(tmp_path: Path):
    search = FakeSearchBackend(fail_first=True)
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Retry this research",
                mode=ResearchMode.DEEP_SEARCH,
                run_id="retry-event-run",
            )
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert len(search.calls) >= 2
    assert result.usage["searches"] >= 2
    events = [
        json.loads(line)
        for line in (Path(result.artifact_dir) / "details" / "runtime" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    event_types = [item["event_type"] for item in events]
    first_review = event_types.index("synthesis_completed")
    researcher_events = [
        index for index, event_type in enumerate(event_types) if event_type == "worker_completed"
    ]
    assert researcher_events[0] < first_review < researcher_events[1]
    planner_traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "planner.json").read_text(encoding="utf-8")
    )
    synthesis_traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "synthesis.json").read_text(encoding="utf-8")
    )
    assert [item["round_no"] for item in planner_traces] == [1, 2]
    assert [item["decision"] for item in synthesis_traces] == ["need_more", "ready"]


@pytest.mark.asyncio
async def test_unified_prompt_roles_receive_runtime_contracts(tmp_path: Path):
    llm = FakeLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Inspect the unified agent contracts",
                constraints={"date_boundary": "截至 2026-07-01"},
            )
        )

    systems = [system for _, system, _ in llm.calls]
    assert any("lead research planner" in item for item in systems)
    assert any("research worker subagent" in item for item in systems)
    assert any("lead research synthesizer" in item for item in systems)
    assert any("final phase of the research synthesizer" in item for item in systems)
    planner_call = next(json.loads(user) for _, system, user in llm.calls if "research planner" in system)
    assert planner_call["date_boundary"] == "截至 2026-07-01"
    worker_traces = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "worker.json").read_text(encoding="utf-8")
    )
    assert {item["phase"] for item in worker_traces} == {"strategy", "reflection"}


def test_legacy_mode_values_normalize_to_standard_mode():
    assert ResearchMode("deep_search") is ResearchMode.STANDARD
    assert ResearchMode("deep_report") is ResearchMode.STANDARD


@pytest.mark.asyncio
async def test_missing_dimension_triggers_single_replan(tmp_path: Path):
    search = FakeSearchBackend()
    service = ResearchService.in_memory(
        llm=FakeLLM(planner_task_limit=1),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    request = ResearchRequest(
        question="Research two dimensions",
        mode=ResearchMode.DEEP_REPORT,
        constraints={"required_dimensions": ["dimension-a", "dimension-b"]},
    )
    async with service:
        result = await service.run(request)

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert len(search.calls) == 2


@pytest.mark.asyncio
async def test_evaluation_replan_uses_gap_closure_prompt_and_existing_requirements(
    tmp_path: Path,
):
    class CapturingPlannerLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.planner_systems: list[str] = []
            self.planner_payloads: list[dict] = []

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "lead research planner" in system:
                self.planner_systems.append(system)
                self.planner_payloads.append(json.loads(user))
            return await super().complete_json(system, user, max_tokens=max_tokens)

    llm = CapturingPlannerLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(fail_first=True),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        await service.run(
            ResearchRequest(
                question="Verify one explicit result",
                profile_id="drb_evaluation",
                constraints={"required_dimensions": ["result"]},
                options={"max_rounds": 2},
            )
        )

    assert len(llm.planner_systems) == 2
    assert "begins with Step 0" in llm.planner_systems[0]
    assert "General rubric-grade planning pattern" in llm.planner_systems[0]
    assert "candidate_cases inventory" in llm.planner_systems[0]
    assert "Complete DAG few-shot" in llm.planner_systems[0]
    assert "neither minimize worker count" in llm.planner_systems[0]
    assert "This is gap closure" in llm.planner_systems[1]
    assert "Example A" not in llm.planner_systems[1]
    assert "coverage_blueprint" in llm.planner_payloads[0]["output_schema"]
    assert "requirements" in llm.planner_payloads[0]["output_schema"]
    assert "requirements" not in llm.planner_payloads[1]["output_schema"]
    assert llm.planner_payloads[1]["existing_requirement_matrix"]


@pytest.mark.asyncio
async def test_planner_expands_candidate_cases_into_worker_owned_field_cells():
    class CandidateMatrixPlannerLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "research planner" not in system:
                return await super().complete_json(system, user, max_tokens=max_tokens)
            return {
                "delivery_contract_required": True,
                "coverage_blueprint": {
                    "repeated_case_fields": ["国家", "样本量", "随访时间"],
                    "requested_tables": [{
                        "title": "研究特征表",
                        "columns": ["研究", "国家", "样本量", "随访时间"],
                        "target_section": "研究特征表",
                    }],
                    "candidate_cases": [
                        {
                            "case_id": "study-alpha",
                            "label": "Alpha Study",
                            "identifiers": ["NCT-ALPHA"],
                            "fields": ["国家", "样本量", "随访时间"],
                        },
                        {
                            "case_id": "study-beta",
                            "label": "Beta Study",
                            "aliases": ["Beta Trial"],
                            "fields": ["国家", "样本量", "随访时间"],
                        },
                    ],
                },
                "requirements": [
                    {"requirement_id": "req-001", "text": "Identify every eligible study."},
                    {"requirement_id": "req-002", "text": "Give the country for every study.", "kind": "table", "field_key": "国家"},
                    {"requirement_id": "req-003", "text": "Give the sample size for every study.", "kind": "table", "field_key": "样本量"},
                    {"requirement_id": "req-004", "text": "Give follow-up for every study.", "kind": "table", "field_key": "随访时间"},
                    {"requirement_id": "req-005", "text": "Explain the overall comparison.", "kind": "analysis"},
                ],
                "tasks": [
                    {
                        "task_key": "origin",
                        "objective": "Discover the candidate inventory.",
                        "research_dimension": "inventory",
                        "source_origin": True,
                        "candidate_case_ids": ["study-alpha", "study-beta"],
                        "requirement_ids": ["req-001"],
                        "queries": ["candidate study inventory"],
                    },
                    {
                        "task_key": "verify-alpha",
                        "objective": "Verify Alpha fields.",
                        "research_dimension": "study characteristics",
                        "candidate_case_ids": ["study-alpha"],
                        "requirement_ids": ["req-002", "req-003", "req-004"],
                        "queries": ["Alpha Study NCT-ALPHA full text"],
                    },
                    {
                        "task_key": "verify-beta",
                        "objective": "Verify Beta fields.",
                        "research_dimension": "study characteristics",
                        "candidate_case_ids": ["study-beta"],
                        "requirement_ids": ["req-002", "req-003", "req-004"],
                        "queries": ["Beta Study full text"],
                    },
                    {
                        "task_key": "analysis",
                        "objective": "Analyze the comparison.",
                        "research_dimension": "comparison",
                        "task_kind": "analyze",
                        "requirement_ids": ["req-005"],
                        "queries": ["overall comparison analysis"],
                    },
                ],
            }, LLMResponse("{}", input_tokens=40, output_tokens=30)

    planner_llm = CandidateMatrixPlannerLLM()
    planner = Planner(planner_llm)
    scope = ResearchScope(
        question="Build a complete study table and analyze it.",
        mode=ResearchMode.STANDARD,
        profile_id="drb_evaluation",
        objective="Verify each study field independently.",
        required_dimensions=["inventory", "study characteristics", "comparison"],
        current_date="2026-08-09",
        options={"max_rounds": 2, "max_workers": 4},
    )
    budget = TaskBudget(max_searches=12, max_fetches=12, max_wall_time_seconds=120)

    first_plan, first_tasks, _, _ = await planner.plan(
        "candidate-matrix-run",
        scope,
        budget,
        round_no=1,
        gaps=[],
        attempted_queries=[],
        evidence_summary={"prior_plans": []},
    )

    requirements = first_plan["requirements"]
    requirement_ids = {item["requirement_id"] for item in requirements}
    assert {"req-002", "req-003", "req-004"}.isdisjoint(requirement_ids)
    assert {"req-001", "req-005"}.issubset(requirement_ids)
    cell_requirements = [item for item in requirements if item.get("case_id")]
    assert len(cell_requirements) == 6
    assert len({item["requirement_id"] for item in cell_requirements}) == 6
    assert {(item["case_id"], item["field_key"]) for item in cell_requirements} == {
        (case_id, field)
        for case_id in ("study-alpha", "study-beta")
        for field in ("国家", "样本量", "随访时间")
    }
    blueprint = first_plan["coverage_blueprint"]
    assert blueprint["case_field_requirement_count"] == 6
    assert blueprint["candidate_inventory_is_untrusted_navigation_data"] is True

    tasks_by_key = {
        task.objective: task
        for task in first_tasks
    }
    origin_task = tasks_by_key["Discover the candidate inventory."]
    alpha_task = tasks_by_key["Verify Alpha fields."]
    beta_task = tasks_by_key["Verify Beta fields."]
    analysis_task = tasks_by_key["Analyze the comparison."]
    assert {case.case_id for case in origin_task.candidate_cases} == {
        "study-alpha", "study-beta"
    }
    assert {case.case_id for case in alpha_task.candidate_cases} == {
        "study-alpha"
    }
    assert {case.case_id for case in beta_task.candidate_cases} == {
        "study-beta"
    }
    assert not analysis_task.candidate_cases
    for task, case_id in ((alpha_task, "study-alpha"), (beta_task, "study-beta")):
        assert len(task.requirement_ids) == 3
        assert all(
            next(item for item in cell_requirements if item["requirement_id"] == requirement_id)["case_id"] == case_id
            for requirement_id in task.requirement_ids
        )

    second_plan, _, _, _ = await planner.plan(
        "candidate-matrix-run",
        scope,
        budget,
        round_no=2,
        gaps=[{"dimension": "study characteristics", "missing_information": "remaining cells"}],
        attempted_queries=[],
        evidence_summary={"prior_plans": [first_plan]},
    )
    first_field_ids = {
        case["case_id"]: case["field_requirement_ids"]
        for case in first_plan["coverage_blueprint"]["candidate_cases"]
    }
    second_field_ids = {
        case["case_id"]: case["field_requirement_ids"]
        for case in second_plan["coverage_blueprint"]["candidate_cases"]
    }
    assert second_field_ids == first_field_ids

    class CapturingStrategyLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.strategy_payload: dict = {}

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            payload = json.loads(user)
            if "research worker subagent" in system and payload.get("phase") == "strategy":
                self.strategy_payload = payload
            return await super().complete_json(system, user, max_tokens=max_tokens)

    strategy_llm = CapturingStrategyLLM()
    worker = ResearchWorker(strategy_llm, SkillRegistry())
    await worker._strategy(
        {
            "run_id": "candidate-matrix-run",
            "scope": {
                "question": scope.question,
                "current_date": scope.current_date,
                "date_boundary": None,
                "options": {"max_worker_iterations": 3},
            },
            "request": {"profile_id": "drb_evaluation"},
        },
        alpha_task,
    )
    projected_cases = strategy_llm.strategy_payload["task_contract"]["candidate_cases"]
    assert [item["case_id"] for item in projected_cases] == ["study-alpha"]


@pytest.mark.asyncio
async def test_replanner_promotes_unverified_candidate_leads_into_field_cells():
    planner = Planner(FakeLLM())
    scope = ResearchScope(
        question="Build a complete study table.",
        mode=ResearchMode.STANDARD,
        profile_id="drb_evaluation",
        objective="Verify every discovered study.",
        required_dimensions=["study characteristics"],
        current_date="2026-08-09",
        options={"max_rounds": 2, "max_workers": 2},
    )
    budget = TaskBudget(max_searches=8, max_fetches=8, max_wall_time_seconds=120)
    prior_plan = {
        "round_no": 1,
        "requirements": [
            {
                "requirement_id": "req-country",
                "text": "Give the country for every study.",
                "kind": "table",
                "field_key": "Country",
            }
        ],
        "coverage_blueprint": {
            "repeated_case_fields": ["Country", "Design"],
            "candidate_cases": [
                {
                    "case_id": "known-study",
                    "label": "Known Study",
                    "fields": ["Country", "Design"],
                }
            ],
        },
    }
    plan, _, _, _ = await planner.plan(
        "lead-promotion-run",
        scope,
        budget,
        round_no=2,
        gaps=[{"dimension": "study characteristics", "missing_information": "remaining cells"}],
        attempted_queries=[],
        evidence_summary={
            "prior_plans": [prior_plan],
            "candidate_leads": [
                {
                    "lead_id": "lead-new-study",
                    "task_id": "origin-task",
                    "title": "New Study",
                    "identifier": "NCT-NEW",
                    "lead_type": "case",
                    "missing_fields": ["Country"],
                    "status": "unverified",
                },
                {
                    "lead_id": "lead-duplicate",
                    "task_id": "origin-task",
                    "title": "Known Study",
                    "identifier": "",
                    "lead_type": "case",
                    "missing_fields": ["Country"],
                    "status": "unverified",
                },
            ],
        },
    )

    cases = plan["coverage_blueprint"]["candidate_cases"]
    assert {item["label"] for item in cases} == {"Known Study", "New Study"}
    new_case = next(item for item in cases if item["label"] == "New Study")
    assert set(new_case["fields"]) == {"Country", "Design"}
    cell_requirements = [
        item for item in plan["requirements"] if item.get("case_id") == new_case["case_id"]
    ]
    assert {item["field_key"] for item in cell_requirements} == {"Country", "Design"}


@pytest.mark.asyncio
async def test_evaluation_replan_adds_tasks_for_uncovered_gaps():
    planner = Planner(FakeLLM(planner_task_limit=1))
    scope = ResearchScope(
        question="Compare national digital strategies",
        mode=ResearchMode.STANDARD,
        profile_id="drb_evaluation",
        objective="Close the remaining comparison gaps",
        required_dimensions=[],
        current_date="2026-08-04",
        options={"max_rounds": 2, "max_workers": 8},
    )
    _, tasks, _, _ = await planner.plan(
        "evaluation-gap-coverage",
        scope,
        TaskBudget(max_searches=24, max_fetches=24, max_wall_time_seconds=120),
        round_no=2,
        gaps=[
            {
                "gap_id": "jp-gap",
                "dimension": "Japan current strategy fields",
                "missing_information": "official date, vision, and pillars",
                "requirement_id": "req-009",
            },
            {
                "gap_id": "sg-gap",
                "dimension": "Singapore current strategy fields",
                "missing_information": "official date, vision, and pillars",
                "requirement_id": "req-019",
            },
            {
                "gap_id": "th-gap",
                "dimension": "Thailand current strategy status",
                "missing_information": "implementation status and hierarchy",
                "requirement_id": "req-023",
            },
        ],
        attempted_queries=[],
        evidence_summary={
            "prior_plans": [
                {
                    "requirements": [
                        {"requirement_id": "req-009", "text": "Japan fields"},
                        {"requirement_id": "req-019", "text": "Singapore fields"},
                        {"requirement_id": "req-023", "text": "Thailand status"},
                    ]
                }
            ]
        },
    )

    assert {task.research_dimension for task in tasks} == {
        "Japan current strategy fields",
        "Singapore current strategy fields",
        "Thailand current strategy status",
    }
    assert all("各国数字化发展历程回顾" not in query for task in tasks for query in task.queries)


@pytest.mark.asyncio
async def test_need_more_without_a_concrete_gap_does_not_trigger_replan(tmp_path: Path):
    class UntargetedReplanLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.planner_calls = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "lead research planner" in system:
                self.planner_calls += 1
            if "lead research synthesizer" in system:
                return (
                    {
                        "decision": "need_more",
                        "reasoning": ["More material might be nice to have."],
                        "coverage": [],
                        "missing_information": [],
                    },
                    LLMResponse("{}", input_tokens=20, output_tokens=10),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    llm = UntargetedReplanLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(question="Answer without an optional replan")
        )

    synthesis = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "synthesis.json").read_text(
            encoding="utf-8"
        )
    )
    assert llm.planner_calls == 1
    assert synthesis[0]["decision"] == "partial"
    assert "verified fact is 42 units" in result.markdown


@pytest.mark.asyncio
async def test_one_replan_does_not_hide_additional_missing_dimensions(tmp_path: Path):
    search = FakeSearchBackend()
    service = ResearchService.in_memory(
        llm=FakeLLM(planner_task_limit=1),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    request = ResearchRequest(
        question="Research three dimensions",
        mode=ResearchMode.DEEP_REPORT,
        run_id="limited-replan-run",
        constraints={"required_dimensions": ["dimension-a", "dimension-b", "dimension-c"]},
    )
    async with service:
        result = await service.run(request)

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert len(search.calls) == 3


@pytest.mark.asyncio
async def test_sqlite_checkpointer_runs_end_to_end(tmp_path: Path):
    service = ResearchService(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "runtime" / "checkpoints.sqlite",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Persist this run",
                mode=ResearchMode.DEEP_SEARCH,
                run_id="sqlite-test-run",
            )
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert (tmp_path / "runtime" / "checkpoints.sqlite").exists()


@pytest.mark.asyncio
async def test_one_fetch_failure_does_not_abort_researcher(tmp_path: Path):
    fetcher = FailOneFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=TwoHitSearchBackend(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(question="Isolate a failed fetch", mode=ResearchMode.DEEP_SEARCH)
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert result.gate.passed
    assert len(fetcher.calls) == 2
    assert result.usage["fetches"] == 2
    assert "Fetch failed" not in result.markdown
    assert "extraction-traces.json" not in result.markdown


@pytest.mark.asyncio
async def test_domain_policy_is_enforced_before_fetch(tmp_path: Path):
    fetcher = FailOneFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=TwoHitSearchBackend(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Use only permitted sources",
                mode=ResearchMode.DEEP_SEARCH,
                constraints={
                    "allowed_domains": ["example.com"],
                    "blocked_domains": ["broken.example.com"],
                },
            )
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert fetcher.calls == ["https://example.com/research"]


@pytest.mark.asyncio
async def test_parallel_tool_timeout_is_isolated_as_partial_result(tmp_path: Path):
    class SelectiveSlowSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            if "risks" in query.lower():
                await asyncio.sleep(0.05)
            return await FakeSearchBackend().search(query, max_results)

    search = SelectiveSlowSearch()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        assert service._graph is not None
        service._graph.worker.tools._tools["web_search"].timeout_seconds = 0.005
        result = await service.run(
            ResearchRequest(
                question="Prepare a timeout-isolated report",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts", "risks", "alternatives"]},
            )
        )

    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert result.sources


@pytest.mark.asyncio
async def test_artifacts_include_replayable_events_snapshots_and_valid_hashes(tmp_path: Path):
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Inspect artifacts",
                mode=ResearchMode.DEEP_SEARCH,
                run_id="artifact-test-run",
            )
        )

    run_dir = Path(result.artifact_dir)
    events = [json.loads(line) for line in (run_dir / "details" / "runtime" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(event["event_type"] == "run_terminal" for event in events) == 1
    assert [event["sequence"] for event in events] == sorted(event["sequence"] for event in events)
    assert json.loads((run_dir / "details" / "evidence" / "source-snapshots.json").read_text(encoding="utf-8"))

    manifest = json.loads((run_dir / "details" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["budget"]["max_total_searches"] == 100
    for name, expected_hash in manifest["artifact_hashes"].items():
        actual_hash = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
    for name in (
        "details/traces/planner.json",
        "details/traces/worker.json",
        "details/traces/synthesis.json",
        "details/traces/search.json",
        "details/traces/fetch.json",
        "details/traces/extraction.json",
        "details/traces/report.json",
        "details/audit/audit-log.json",
        "details/audit/audit-report.md",
    ):
        assert (run_dir / name).exists()
    assert sorted(item.name for item in run_dir.iterdir()) == ["details", "report.md"]
    audit = json.loads((run_dir / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8"))
    assert audit["workflow_events"]
    assert audit["task_attempts"]
    assert audit["llm_outputs"]["planner"]
    assert audit["tool_outputs"]["search"]
    assert audit["tool_outputs"]["fetch"]
    assert "[研究审计报告](details/audit/audit-report.md)" in result.markdown
    assert "[结构化审计日志](details/audit/audit-log.json)" in result.markdown
    assert result.audit_report_path == str(run_dir / "details" / "audit" / "audit-report.md")
    assert result.audit_log_path == str(run_dir / "details" / "audit" / "audit-log.json")


@pytest.mark.asyncio
async def test_report_continues_after_multiple_fetch_failures_until_floor_passes(tmp_path: Path):
    class RedundantSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            urls = [
                "https://broken-one.example/source",
                "https://broken-two.example/source",
                "https://verified-one.example/source",
                "https://verified-two.example/source",
                "https://independent.example.net/source",
            ]
            return [
                {
                    "title": f"Source {index}",
                    "url": url,
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": index,
                }
                for index, url in enumerate(urls, 1)
            ]

    class FailFirstTwoFetcher(FakeFetcher):
        async def fetch(self, url: str):
            self.calls.append(url)
            if len(self.calls) <= 2:
                raise RuntimeError("injected early failure")
            return await FakeFetcher().fetch(url)

    fetcher = FailFirstTwoFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=RedundantSearch(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Research despite early source failures",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
                options={
                    "task_sufficiency": {
                        "min_successful_extractions": 2,
                        "min_effective_sources": 2,
                        "min_independent_domains": 2,
                        "min_evidence": 2,
                    }
                },
            )
        )

    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8")
    )
    contribution = audit["task_sufficiency"][0]
    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert [item["status"] for item in audit["tool_outputs"]["fetch"]][:2] == [
        "failed",
        "failed",
    ]
    assert contribution["successful_extractions"] >= 2
    assert contribution["sufficiency_passed"]


@pytest.mark.asyncio
async def test_valid_fetch_with_zero_accepted_claims_does_not_count_as_extraction(tmp_path: Path):
    class EmptyExtractionLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "final research writer" in system:
                return (
                    {
                        "sections": [
                            {
                                "title": "Writer-led answer",
                                "paragraphs": [
                                    {
                                        "text": "The Writer can still provide a useful answer from its own knowledge.",
                                        "claim_ids": [],
                                    }
                                ],
                            }
                        ],
                        "missing_dimensions": [],
                    },
                    LLMResponse("{}", input_tokens=20, output_tokens=10),
                )
            if "research planner" in system:
                return await super().complete_json(system, user, max_tokens=max_tokens)
            self.calls.append(("json", system, user))
            return (
                {"claims": [], "knowledge_gaps": []},
                LLMResponse("{}", input_tokens=20, output_tokens=5),
            )

    service = ResearchService.in_memory(
        llm=EmptyExtractionLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(question="Reject empty extraction", mode=ResearchMode.DEEP_SEARCH)
        )

    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8")
    )
    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert "Writer can still provide a useful answer" in result.markdown
    assert audit["task_sufficiency"][0]["successful_extractions"] == 0
    assert all(item["status"] == "no_evidence" for item in audit["tool_outputs"]["fetch"])


@pytest.mark.asyncio
async def test_repeated_pages_on_one_domain_do_not_meet_domain_floor(tmp_path: Path):
    class OneDomainSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            call = len(self.calls)
            return [
                {
                    "title": f"Same domain {call}-{index}",
                    "url": f"https://sub{index}.same.example/source-{call}-{index}",
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": index,
                }
                for index in range(1, 3)
            ]

    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=OneDomainSearch(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Require genuinely independent sources",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
                options={
                    "task_sufficiency": {
                        "min_effective_sources": 2,
                        "min_independent_domains": 2,
                    }
                },
            )
        )

    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8")
    )
    contribution = audit["task_sufficiency"][-1]
    assert result.status == RunStatus.COMPLETED_PARTIAL
    # A same-domain follow-up no longer burns all remaining rounds. Round 2
    # adds pages but neither closes a high-priority gap nor adds an independent
    # source, so the graph returns a useful partial result.
    assert contribution["attempt"] == 2
    assert max(
        item["independent_domain_count"] for item in audit["task_sufficiency"]
    ) == 1
    assert not contribution["sufficiency_passed"]


@pytest.mark.asyncio
async def test_parallel_workers_share_one_in_flight_url_fetch(tmp_path: Path):
    class SameUrlSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            return [
                {
                    "title": "Shared primary source",
                    "url": "https://shared.example/source",
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": 1,
                }
            ]

    class SlowCountingFetcher(FakeFetcher):
        async def fetch(self, url: str):
            self.calls.append(url)
            await asyncio.sleep(0.03)
            return await FakeFetcher().fetch(url)

    fetcher = SlowCountingFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=SameUrlSearch(),
        fetcher=fetcher,
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Share duplicate network work across workers",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts", "risks"]},
                options={"max_rounds": 1},
            )
        )

    assert result.sources
    assert fetcher.calls == ["https://shared.example/source"]
    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(
            encoding="utf-8"
        )
    )
    reuse_modes = {
        item.get("reuse_mode")
        for item in audit["tool_outputs"]["fetch"]
    }
    assert "shared" in reuse_modes


@pytest.mark.asyncio
async def test_partial_sufficiency_override_preserves_report_defaults_and_attempt_limit(
    tmp_path: Path,
):
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Require more evidence than the fixture can provide",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
                options={
                    "task_sufficiency": {"min_evidence": 7},
                    "max_rounds": 3,
                },
            )
        )

    plan = json.loads(
        (Path(result.artifact_dir) / "details" / "planning" / "research-plan.json").read_text(encoding="utf-8")
    )
    task = plan["tasks"][0]
    assert len(plan["plans"]) == 2
    assert task["sufficiency"] == {
        "min_queries": 1,
        "min_query_coverage": 1,
        "min_successful_extractions": 1,
        "min_effective_sources": 1,
        "min_independent_domains": 1,
        "min_evidence": 7,
        "max_source_quality_tier": 3,
    }
    assert result.status == RunStatus.COMPLETED_PARTIAL


@pytest.mark.asyncio
async def test_planner_queries_are_preserved_when_three_distinct_queries_are_returned(
    tmp_path: Path,
):
    class ThreeQueryPlannerLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "research planner" in system:
                return (
                    {
                        "tasks": [
                            {
                                "objective": "Investigate facts",
                                "research_dimension": "facts",
                                "queries": ["raw-query-a", "raw-query-b", "raw-query-c"],
                            }
                        ]
                    },
                    LLMResponse("{}", input_tokens=20, output_tokens=10),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    search = FakeSearchBackend()
    service = ResearchService.in_memory(
        llm=ThreeQueryPlannerLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        await service.run(
            ResearchRequest(
                question="Keep planner queries",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
            )
        )

    assert search.calls[:3] == ["raw-query-a", "raw-query-b", "raw-query-c"]


@pytest.mark.asyncio
async def test_verified_not_yet_available_status_stops_research_loop(tmp_path: Path):
    class AvailabilityAwareLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "research synthesizer" in system and "final research writer" not in system:
                payload = json.loads(user)
                claim_id = payload["validated_claim_ledger"][0]["claim_id"]
                return (
                    {
                        "decision": "need_more",
                        "answer_status": "not_yet_available",
                        "answer_summary": "The requested result is scheduled for later.",
                        "answer_supporting_claim_ids": [claim_id, "clm_not_in_ledger"],
                        "reasoning": ["A verified schedule establishes current availability."],
                        "coverage": [],
                        "missing_information": [
                            {
                                "dimension": "future result",
                                "missing_information": "result",
                                "priority": "high",
                            }
                        ],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=20),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=AvailabilityAwareLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(ResearchRequest(question="What is the future result?"))

    plan = json.loads(
        (Path(result.artifact_dir) / "details" / "planning" / "research-plan.json").read_text(encoding="utf-8")
    )
    synthesis = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "synthesis.json").read_text(encoding="utf-8")
    )
    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert len(plan["plans"]) == 1
    assert synthesis[0]["decision"] == "ready"
    assert "42 units" in result.markdown


@pytest.mark.asyncio
async def test_writer_cannot_override_ready_with_self_declared_gaps(
    tmp_path: Path,
):
    class GapInventingWriterLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "final research writer" in system:
                payload = json.loads(user)
                claim_id = payload["claim_ledger"][0]["claim_id"]
                return (
                    {
                        "sections": [
                            {
                                "title": "核心发现",
                                "paragraphs": [
                                    {
                                        "text": "The verified fact is 42 units.",
                                        "claim_ids": [claim_id],
                                    }
                                ],
                            }
                        ],
                        "missing_dimensions": [
                            {"dimension": "facts", "reason": "Other facts unavailable."}
                        ],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=10),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=GapInventingWriterLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Do not invent gaps",
                mode=ResearchMode.DEEP_SEARCH,
            )
        )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert "Other facts unavailable" not in result.markdown
    assert "42 units" in result.markdown


@pytest.mark.asyncio
async def test_writer_relies_on_llm_synthesis_without_numeric_surface_gate(
    tmp_path: Path,
):
    class RepairingWriterLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.writer_calls = 0
            self.reviewer_calls = 0
            self.writer_system = ""

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "final report reviewer" in system:
                self.reviewer_calls += 1
            if "final research writer" in system:
                self.writer_calls += 1
                self.writer_system = system
                payload = json.loads(user)
                claim_id = payload["claim_ledger"][0]["claim_id"]
                return (
                    {
                        "sections": [
                            {
                                "title": "核心发现",
                                "paragraphs": [
                                    {
                                        "text": "The localized synthesis says 99 units.",
                                        "claim_ids": [claim_id],
                                    }
                                ],
                            }
                        ],
                        "missing_dimensions": [],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=10),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    llm = RepairingWriterLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(ResearchRequest(question="Repair invalid writer numbers"))

    assert llm.writer_calls == 1
    assert llm.reviewer_calls == 0
    assert "not the boundary of your knowledge" in llm.writer_system
    assert "split it into separate" in llm.writer_system
    assert "not proof that a" in llm.writer_system
    assert "qualify only the detail" in llm.writer_system
    assert "99 units" in result.markdown


@pytest.mark.asyncio
async def test_general_profile_shares_quality_rules_without_evaluation_bookkeeping(
    tmp_path: Path,
):
    llm = FakeLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        await service.run(
            ResearchRequest(
                question="Compare the named items in a table",
                profile_id="general_research",
                options={"max_rounds": 1},
            )
        )

    planner_system, planner_user = next(
        (system, user)
        for _, system, user in llm.calls
        if "lead research planner" in system
    )
    worker_system = next(
        system for _, system, _ in llm.calls if "research worker subagent" in system
    )
    synthesizer_system, synthesizer_user = next(
        (system, user)
        for _, system, user in llm.calls
        if "lead research synthesizer" in system
    )
    writer_system, writer_user = next(
        (system, user)
        for _, system, user in llm.calls
        if "final research writer" in system
    )
    planner_payload = json.loads(planner_user)
    synthesizer_payload = json.loads(synthesizer_user)
    writer_payload = json.loads(writer_user)

    assert "internal coverage check" in planner_system
    assert "requirement matrix" not in planner_payload["output_schema"]
    assert "delivery_contract_required" in planner_payload["output_schema"]
    assert "requirements" in planner_payload["output_schema"]
    assert "requirement_ids" in planner_payload["output_schema"]["tasks"][0]
    assert planner_payload["delivery_contract_policy"]["soft"] is True
    assert "evaluation_profile" not in planner_payload
    assert "coverage ledger" in worker_system
    assert "requirement_ids assigned" not in worker_system
    assert "another targeted" in synthesizer_system
    assert "requirements" not in synthesizer_payload
    assert "evaluation_profile" not in synthesizer_payload
    assert "not the boundary of your knowledge" in writer_system
    assert "not proof that a" in writer_system
    assert "refusing an entire section" in writer_payload["rules"][2]
    assert writer_payload["output_schema"]["sections"][0]["tables"]
    assert "requirements" not in writer_payload
    assert "evaluation_profile" not in writer_payload
    assert "requirement_ids" not in writer_payload["output_schema"]["sections"][0][
        "paragraphs"
    ][0]
    assert "requirement_ids" not in writer_payload["output_schema"]["sections"][0][
        "tables"
    ][0]
    assert not any("final report reviewer" in system for _, system, _ in llm.calls)


@pytest.mark.asyncio
async def test_general_structured_request_uses_soft_contract_and_on_demand_review(
    tmp_path: Path,
):
    class ContractLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.synthesizer_payload: dict = {}
            self.writer_payload: dict = {}
            self.reviewer_calls = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            payload = json.loads(user)
            if "lead research planner" in system:
                return {
                    "delivery_contract_required": True,
                    "query_type": "breadth_first",
                    "research_strategy": "cover both requested rows and fields",
                    "answer_requirements": ["Compare Alpha and Beta"],
                    "coverage_blueprint": {
                        "requested_output_shapes": ["table"],
                        "independent_coverage_axes": ["item"],
                        "explicit_items": ["Alpha", "Beta"],
                        "repeated_case_fields": ["Value", "Meaning"],
                        "requested_tables": [
                            {
                                "title": "Comparison",
                                "columns": ["Item", "Value", "Meaning"],
                                "row_labels": ["Alpha", "Beta"],
                                "target_section": "Comparison",
                            }
                        ],
                        "grouping_rationale": "one coherent comparison task",
                    },
                    "requirements": [
                        {
                            "text": "Give Alpha's value",
                            "kind": "number",
                            "target_section": "Comparison",
                            "item_key": "Alpha",
                            "field_key": "Value",
                            "importance": "required",
                            "display_order": 1,
                        },
                        {
                            "text": "Explain optional background",
                            "kind": "analysis",
                            "target_section": "Comparison",
                            "importance": "preferred",
                            "display_order": 2,
                        },
                    ],
                    "tasks": [
                        {
                            "task_key": "comparison",
                            "objective": "Verify the requested comparison fields",
                            "research_dimension": "comparison",
                            "research_object": "Alpha and Beta comparison",
                            "task_kind": "gather",
                            "depends_on": [],
                            "produces_requirement_ids": ["req-001", "req-002"],
                            "requirement_ids": ["req-001", "req-002"],
                            "queries": ["Alpha Beta comparison value"],
                            "key_questions": ["What values are supported?"],
                            "preferred_source_types": ["primary"],
                            "verification_strategy": "verify the source text",
                            "expected_output": "supported comparison fields",
                            "sufficiency": {
                                "min_queries": 1,
                                "min_query_coverage": 1,
                                "min_successful_extractions": 1,
                                "min_effective_sources": 1,
                                "min_independent_domains": 1,
                                "min_evidence": 1,
                            },
                        }
                    ],
                }, LLMResponse("{}", input_tokens=20, output_tokens=20)
            if "lead research synthesizer" in system:
                self.synthesizer_payload = payload
            if "final research writer" in system:
                self.writer_payload = payload
            if "final report reviewer" in system:
                self.reviewer_calls += 1
                return {
                    "decision": "pass",
                    "issues": [],
                }, LLMResponse("{}", input_tokens=10, output_tokens=5)
            return await super().complete_json(system, user, max_tokens=max_tokens)

    llm = ContractLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        await service.run(
            ResearchRequest(
                question="Compare Alpha and Beta in a table with value and meaning",
                profile_id="general_research",
                options={"max_rounds": 1},
            )
        )

    assert [item["importance"] for item in llm.synthesizer_payload["requirements"]] == [
        "required",
        "preferred",
    ]
    assert llm.synthesizer_payload["delivery_contract"]["soft"] is True
    assert "coverage_blueprint" not in llm.writer_payload
    assert "dual_coverage" not in llm.writer_payload
    assert llm.writer_payload["synthesis_decision"]["coverage"] == []
    assert llm.writer_payload["requested_output_structure"]["requested_tables"][0][
        "columns"
    ] == ["Item", "Value", "Meaning"]
    assert llm.writer_payload["delivery_contract"]["missing_material_is_non_blocking"] is True
    assert llm.reviewer_calls == 1


@pytest.mark.asyncio
async def test_evaluation_profile_applies_llm_report_review(tmp_path: Path):
    class ReviewingLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.reviewer_calls = 0
            self.planner_system = ""
            self.synthesizer_system = ""
            self.writer_system = ""
            self.reviewer_system = ""
            self.writer_payload: dict = {}
            self.reviewer_payload: dict = {}

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            payload = json.loads(user)
            if "final report reviewer" in system:
                self.reviewer_calls += 1
                self.reviewer_system = system
                self.reviewer_payload = payload
                if self.reviewer_calls > 1:
                    return (
                        {"decision": "pass", "issues": [], "revised_draft": None},
                        LLMResponse("{}", input_tokens=20, output_tokens=10),
                    )
                return (
                    {
                        "decision": "repair",
                        "issues": [
                            {
                                "location": "table row 1",
                                "severity": "medium",
                                "issue_type": "grounding",
                                "message": "Repaired the row semantically.",
                            }
                        ],
                        "revised_draft": None,
                    },
                    LLMResponse("{}", input_tokens=25, output_tokens=15),
                )
            if "final research writer" in system:
                if "current_draft" in payload:
                    draft = payload["current_draft"]
                    draft["sections"][0]["tables"][0]["rows"][0][0]["text"] = (
                        "Reviewed entity"
                    )
                    return (
                        draft,
                        LLMResponse("{}", input_tokens=28, output_tokens=18),
                    )
                self.writer_system = system
                self.writer_payload = payload
                claim_id = payload["claim_ledger"][0]["claim_id"]
                return (
                    {
                        "sections": [
                            {
                                "title": "Comparison",
                                "paragraphs": [],
                                "tables": [
                                    {
                                        "title": "Results",
                                        "columns": ["Entity", "Description"],
                                        "rows": [
                                            [
                                                {
                                                    "text": "Writer entity",
                                                    "cell_type": "factual",
                                                    "claim_ids": [],
                                                },
                                                {
                                                    "text": "The verified fact is 42 units.",
                                                    "cell_type": "factual",
                                                    "claim_ids": [claim_id],
                                                },
                                            ]
                                        ],
                                    }
                                ],
                            }
                        ],
                        "missing_dimensions": [],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=20),
                )
            if "lead research planner" in system:
                self.planner_system = system
            if "lead research synthesizer" in system:
                self.synthesizer_system = system
            return await super().complete_json(system, user, max_tokens=max_tokens)

    llm = ReviewingLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Compare the verified result in a table",
                profile_id="drb_evaluation",
                options={"max_rounds": 1},
            )
        )

    report_trace = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert llm.reviewer_calls == 2
    assert "Reviewed entity" in result.markdown
    assert report_trace["review_applied"] is True
    assert report_trace["review_passed"] is True
    assert report_trace["review_error"] is None
    assert report_trace["review_input_tokens"] == 25
    assert report_trace["rewrite_input_tokens"] == 28
    assert report_trace["recheck_input_tokens"] == 20
    assert result.usage["input_tokens"] >= 103
    assert "must be atomic" in llm.planner_system
    assert "another targeted" in llm.synthesizer_system
    assert "not the boundary of your knowledge" in llm.writer_system
    assert "retrieval coverage" in llm.writer_system
    assert "qualified best-effort answer" in llm.writer_system
    assert "must not be rejected merely" in llm.reviewer_system
    assert "Repair blanket refusals" in llm.reviewer_system
    assert "Do not turn uncertain synthesis" in llm.reviewer_system
    assert "Treat deletion as an exceptional last" in llm.reviewer_system
    assert "do not delete useful content solely because it is" in llm.reviewer_system
    assert llm.writer_payload["evaluation_profile"]["allow_derived_analysis"] is True
    assert any("not a knowledge boundary" in rule for rule in llm.writer_payload["rules"])
    assert llm.reviewer_payload["evaluation_profile"]["allow_derived_analysis"] is True
    assert "analysis_depth" in llm.reviewer_payload["output_schema"]["issues"][0][
        "issue_type"
    ]


@pytest.mark.asyncio
async def test_evaluation_profile_keeps_writer_draft_when_review_fails(tmp_path: Path):
    class FailingReviewerLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "final report reviewer" in system:
                raise RuntimeError("review unavailable")
            if "final research writer" in system:
                payload = json.loads(user)
                claim_id = payload["claim_ledger"][0]["claim_id"]
                return (
                    {
                        "sections": [
                            {
                                "title": "Result",
                                "paragraphs": [
                                    {
                                        "text": "Keep the Writer draft: 42 units.",
                                        "claim_ids": [claim_id],
                                    }
                                ],
                            }
                        ],
                        "missing_dimensions": [],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=20),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=FailingReviewerLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Keep a supported answer when review is unavailable",
                profile_id="drb_evaluation",
                options={"max_rounds": 1},
            )
        )

    report_trace = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "Keep the Writer draft" in result.markdown
    assert report_trace["review_applied"] is False
    assert "review unavailable" in report_trace["review_error"]
    assert report_trace["fallback_used"] is False


@pytest.mark.asyncio
async def test_evaluation_profile_lets_writer_restore_missing_requested_section(
    tmp_path: Path,
):
    class MissingSectionReviewerLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.reviewer_calls = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "final report reviewer" in system:
                payload = json.loads(user)
                self.reviewer_calls += 1
                if self.reviewer_calls > 1:
                    return (
                        {"decision": "pass", "issues": [], "revised_draft": None},
                        LLMResponse("{}", input_tokens=20, output_tokens=10),
                    )
                return (
                    {
                        "decision": "repair",
                        "issues": [
                            {
                                "location": "Part B",
                                "severity": "high",
                                "issue_type": "structure",
                                "message": "The requested Part B section is missing.",
                            }
                        ],
                        "revised_draft": None,
                    },
                    LLMResponse("{}", input_tokens=25, output_tokens=15),
                )
            if "final research writer" in system:
                payload = json.loads(user)
                if "current_draft" in payload:
                    draft = payload["current_draft"]
                    draft["sections"].append(
                        {
                            "title": "Part B",
                            "paragraphs": [
                                {
                                    "text": "Part B restores the requested analysis.",
                                    "claim_ids": [],
                                }
                            ],
                            "tables": [],
                        }
                    )
                    return (
                        draft,
                        LLMResponse("{}", input_tokens=28, output_tokens=18),
                    )
                claim_id = payload["claim_ledger"][0]["claim_id"]
                return (
                    {
                        "sections": [
                            {
                                "title": "Part A",
                                "paragraphs": [
                                    {
                                        "text": "Part A keeps the verified 42 units.",
                                        "claim_ids": [claim_id],
                                    }
                                ],
                            },
                        ],
                        "missing_dimensions": [],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=20),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=MissingSectionReviewerLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Write Part A and Part B as separate report sections",
                profile_id="drb_evaluation",
                options={"max_rounds": 1},
            )
        )

    report_trace = json.loads(
        (Path(result.artifact_dir) / "details" / "traces" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "Part A" in result.markdown
    assert "Part B" in result.markdown
    assert report_trace["review_applied"] is True
    assert report_trace["review_passed"] is True
    assert report_trace["review_error"] is None


@pytest.mark.asyncio
async def test_partial_report_uses_writer_localized_evidence_limit(tmp_path: Path):
    class LocalizedGapWriterLLM(FakeLLM):
        async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
            if "research synthesizer" in system and "final research writer" not in system:
                payload = json.loads(user)
                claim_id = payload["validated_claim_ledger"][0]["claim_id"]
                return (
                    {
                        "decision": "need_more",
                        "answer_status": "insufficient",
                        "answer_summary": "More evidence is needed.",
                        "answer_supporting_claim_ids": [claim_id],
                        "reasoning": ["One required dimension is incomplete."],
                        "coverage": [],
                        "missing_information": [
                            {
                                "dimension": "English internal dimension",
                                "missing_information": "English internal gap",
                                "priority": "high",
                            }
                        ],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=20),
                )
            if "final research writer" in system:
                payload = json.loads(user)
                claim_id = payload["claim_ledger"][0]["claim_id"]
                return (
                    {
                        "sections": [
                            {
                                "title": "核心发现",
                                "paragraphs": [
                                    {
                                        "text": "The verified fact is 42 units.",
                                        "claim_ids": [claim_id],
                                    }
                                ],
                            }
                        ],
                        "missing_dimensions": [
                            {"dimension": "证据范围", "reason": "关键对比数据仍待核验。"}
                        ],
                    },
                    LLMResponse("{}", input_tokens=30, output_tokens=10),
                )
            return await super().complete_json(system, user, max_tokens=max_tokens)

    service = ResearchService.in_memory(
        llm=LocalizedGapWriterLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Write a localized partial report",
                options={"max_rounds": 1},
            )
        )

    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert "关键对比数据仍待核验" not in result.markdown
    structured_output = json.loads(
        (Path(result.artifact_dir) / "details/output/structured-output.json").read_text(
            encoding="utf-8"
        )
    )
    assert structured_output["unresolved_gaps"] == ["关键对比数据仍待核验。"]
    assert "English internal" not in result.markdown


@pytest.mark.asyncio
async def test_tier_four_source_is_audited_but_does_not_satisfy_source_floor(
    tmp_path: Path,
):
    class MixedQualitySearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            urls = [
                "https://www.youtube.com/watch?v=untrusted",
                "https://verified.example.org/source",
                "https://www.douyin.com/search/fixture",
            ]
            return [
                {
                    "title": f"Source {index}",
                    "url": url,
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": index,
                }
                for index, url in enumerate(urls, 1)
            ]

    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=MixedQualitySearch(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Do not count low-control pages toward sufficiency",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
                options={
                    "task_sufficiency": {
                        "min_effective_sources": 2,
                        "min_independent_domains": 2,
                    }
                },
            )
        )

    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8")
    )
    contribution = audit["task_sufficiency"][0]
    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert len(audit["ledger"]["sources"]) == 3
    assert contribution["effective_source_count"] == 1
    youtube = next(
        item for item in audit["ledger"]["sources"] if "youtube.com" in item["canonical_uri"]
    )
    assert youtube["quality_tier"] == 4


@pytest.mark.asyncio
async def test_tier_four_only_claim_is_published_as_partial_with_quality_metadata(tmp_path: Path):
    class SocialOnlySearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            return [
                {
                    "title": "Social result",
                    "url": "https://www.youtube.com/watch?v=only-source",
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": 1,
                }
            ]

    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=SocialOnlySearch(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(question="Reject low-control-only evidence", mode=ResearchMode.DEEP_SEARCH)
        )

    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8")
    )
    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert audit["ledger"]["claims"]
    assert "42 units" in result.markdown
    assert "youtube.com" in result.markdown


@pytest.mark.asyncio
async def test_multiple_snapshots_of_one_canonical_url_count_as_one_source(
    tmp_path: Path,
):
    class RedirectAliasSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            return [
                {
                    "title": f"Alias {index}",
                    "url": f"https://alias-{index}.example/source",
                    "snippet": "The verified fact is 42 units.",
                    "provider": self.name,
                    "rank": index,
                }
                for index in range(1, 3)
            ]

    class CanonicalizingFetcher(FakeFetcher):
        async def fetch(self, url: str):
            self.calls.append(url)
            return {
                "requested_url": url,
                "final_url": "https://canonical.example/source",
                "title": "Canonical source",
                "content": (
                    "The verified fact is 42 units. Additional context is available. "
                    + url
                ),
                "content_type": "text/plain",
                "http_status": 200,
            }

    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=RedirectAliasSearch(),
        fetcher=CanonicalizingFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(
                question="Do not double count canonical URLs",
                mode=ResearchMode.DEEP_REPORT,
                constraints={"required_dimensions": ["facts"]},
                options={
                    "task_sufficiency": {
                        "min_effective_sources": 2,
                        "min_independent_domains": 2,
                    }
                },
            )
        )

    audit = json.loads(
        (Path(result.artifact_dir) / "details" / "audit" / "audit-log.json").read_text(encoding="utf-8")
    )
    contribution = audit["task_sufficiency"][0]
    assert result.status == RunStatus.COMPLETED_PARTIAL
    assert len(audit["ledger"]["sources"]) == 2
    assert contribution["effective_source_count"] == 1
    assert contribution["independent_domain_count"] == 1


@pytest.mark.asyncio
async def test_sqlite_resume_continues_failed_persist_without_researching_again(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "runtime" / "resume.sqlite"
    artifact_root = tmp_path / "artifacts"
    first_search = FakeSearchBackend()
    first = ResearchService(
        llm=FakeLLM(),
        search_backend=first_search,
        fetcher=FakeFetcher(),
        artifact_root=artifact_root,
        checkpoint_path=checkpoint_path,
    )
    async with first:
        assert first._graph is not None

        def fail_persist(state):
            raise RuntimeError("injected artifact failure")

        first._graph.workspace.persist_run = fail_persist
        with pytest.raises(RuntimeError, match="injected artifact failure"):
            await first.run(
                ResearchRequest(
                    question="Resume after persistence failure",
                    mode=ResearchMode.DEEP_SEARCH,
                    run_id="interrupted-resume-run",
                )
            )
    assert len(first_search.calls) == 1

    second_search = FakeSearchBackend()
    second = ResearchService(
        llm=FakeLLM(),
        search_backend=second_search,
        fetcher=FakeFetcher(),
        artifact_root=artifact_root,
        checkpoint_path=checkpoint_path,
    )
    async with second:
        result = await second.resume("interrupted-resume-run")
        with pytest.raises(RunAlreadyExistsError):
            await second.run(
                ResearchRequest(
                    question="Do not restart this run",
                    mode=ResearchMode.DEEP_SEARCH,
                    run_id="interrupted-resume-run",
                )
            )

    assert result.status == RunStatus.COMPLETED_VERIFIED
    assert Path(result.output_path).exists()
    assert second_search.calls == []
    assert result.usage["searches"] == 1
    events = [
        json.loads(line)
        for line in (Path(result.artifact_dir) / "details" / "runtime" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(item["event_type"] == "run_terminal" for item in events) == 1
