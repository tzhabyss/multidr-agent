import asyncio
import json
import time

import pytest

from deepresearch.agents import (
    ResearchWorker,
    _candidate_meets_quality_floor,
    _diversify_hits,
    _expand_queries,
    _gap_query_candidates,
    _is_generic_replan_query,
    _prior_search_candidate_groups,
    _subject_from_question,
    _time_aware_fallback_query,
)
from deepresearch.graph import (
    ResearchGraph,
    _has_unattempted_search_candidates,
    _required_next_round_seconds,
)
from deepresearch.schemas import FetchedDocument, ResearchTask, SearchHit, TaskBudget
from deepresearch.skill_registry import SkillDefinition, SkillRegistry
from deepresearch.llm import LLMResponse


def test_subject_extraction_prefers_quoted_research_topic():
    question = (
        "\u8bf7\u4ee5\u201c\u6784\u5efa\u6709\u6548\u5206\u6563\u98ce\u9669\u9700\u8981\u591a\u5c11\u53ea\u80a1\u7968\uff1f\u201d"
        "\u4e3a\u4e3b\u9898\u64b0\u5199\u8c03\u7814\u62a5\u544a"
    )

    assert _subject_from_question(question) == (
        "\u6784\u5efa\u6709\u6548\u5206\u6563\u98ce\u9669\u9700\u8981\u591a\u5c11\u53ea\u80a1\u7968"
    )


def test_query_expansion_does_not_embed_the_entire_instruction():
    question = "请研究 DeepResearch Bench，并输出十个维度、逐项引用、最后给出表格。"

    queries = _expand_queries(question, "信息召回", [question], minimum=2)

    assert all(question not in query for query in queries)
    assert all(len(query) <= 180 for query in queries)
    assert any("DeepResearch Bench" in query for query in queries)


def test_gap_queries_replace_prose_replan_seeds_with_retrieval_terms():
    gap = {
        "dimension": "韩国数字政府总体规划具体方向",
        "missing_information": "完整战略支柱和主要实施方向",
        "preferred_source_types": ["韩国行政安全部官方规划"],
    }

    queries = _gap_query_candidates(
        gap["dimension"],
        gap,
        attempted={"韩国数字政府总体规划具体方向 完整战略支柱和主要实施方向 official"},
    )

    assert queries
    assert all(len(query) <= 180 for query in queries)
    assert all("各国数字化发展历程回顾" not in query for query in queries)
    assert any("official" in query for query in queries)
    assert _is_generic_replan_query('"各国数字化发展历程回顾" 韩国字段')
    assert not _is_generic_replan_query("Korea Digital Government master plan pillars official")


def test_gap_query_candidates_preserve_explicit_historical_boundary():
    queries = _gap_query_candidates(
        "东京逐日降水预报",
        {"missing_information": "逐日降水概率"},
        date_boundary="截至 2024-04-30",
    )

    assert any("2024" in query for query in queries)
    assert all("2022" not in query for query in queries)
    assert all("最新" not in query and "latest" not in query.casefold() for query in queries)


def test_time_aware_fallback_recognizes_compact_chinese_date_boundary():
    query = _time_aware_fallback_query(
        "请核实东京天气",
        "逐日降水预报",
        date_boundary="截至2024年4月30日",
    )

    assert "2024" in query
    assert "最新" not in query


def test_time_aware_fallback_does_not_turn_historical_request_into_latest():
    historical = _time_aware_fallback_query(
        "请核实东京天气",
        "逐日降水预报",
        date_boundary="截至 2024-04-30",
    )
    current = _time_aware_fallback_query("请核实东京天气", "逐日降水预报")

    assert "2024" in historical
    assert "最新" not in historical
    assert "最新" in current


def test_candidate_dedup_preserves_search_order_for_worker_selection():
    groups = [
        [
            (
                "query-1",
                SearchHit(
                    title="Publisher",
                    url="https://www.sciencedirect.com/science/article/pii/example",
                    provider="test",
                ),
            ),
            (
                "query-1",
                SearchHit(
                    title="Open index",
                    url="https://www.semanticscholar.org/paper/example",
                    provider="test",
                ),
            ),
        ]
    ]

    ranked = _diversify_hits(groups)

    assert [item[1].title for item in ranked] == ["Publisher", "Open index"]


def test_evaluation_candidate_floor_skips_low_control_platforms():
    low_control = SearchHit(
        title="User post",
        url="https://medium.com/example/unverified-post",
        provider="test",
    )

    assert not _candidate_meets_quality_floor(
        low_control,
        enforce_quality_floor=True,
        max_source_quality_tier=3,
    )


def test_general_profile_can_retain_low_control_fallback_candidates():
    low_control = SearchHit(
        title="User post",
        url="https://www.slideshare.net/example/deck",
        provider="test",
    )

    assert _candidate_meets_quality_floor(
        low_control,
        enforce_quality_floor=False,
        max_source_quality_tier=3,
    )


@pytest.mark.asyncio
async def test_worker_receives_complete_coarse_candidate_inventory_and_selects_order():
    class SelectionLLM:
        model_id = "selection-model"

        def __init__(self):
            self.payload = {}

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            self.payload = json.loads(user)
            candidate_ids = [
                item["candidate_id"] for item in self.payload["candidates"]
            ]
            return {
                "selected_candidate_ids": list(reversed(candidate_ids)),
                "next_action": "fetch_selected",
                "rationale": "reverse order proves Worker control",
            }, LLMResponse("{}", input_tokens=100, output_tokens=20)

    llm = SelectionLLM()
    worker = ResearchWorker(llm, SkillRegistry())
    task = ResearchTask(
        task_id="task-selection",
        objective="Choose useful sources",
        research_dimension="source selection",
        research_object="selection subject",
        queries=["selection subject"],
        budget=TaskBudget(max_searches=2, max_fetches=30, max_wall_time_seconds=30),
    )
    candidates = [
        (
            [f"query-{index}"],
            SearchHit(
                title=f"Candidate {index}",
                url=f"https://source-{index}.example/research",
                snippet=f"Complete snippet {index}",
                provider="test",
                rank=index,
            ),
        )
        for index in range(25)
    ]

    selected, trace, _, action = await worker._select_candidates(
        {
            "run_id": "run-selection",
            "scope": {},
            "claims": [],
            "unresolved_gaps": [],
        },
        task,
        candidates,
        profile_id="general_research",
        selection_no=1,
        remaining_fetches=30,
        query_texts={f"query-{index}": f"query text {index}" for index in range(25)},
    )

    assert len(llm.payload["candidates"]) == 25
    assert llm.payload["candidates"][-1]["snippet"] == "Complete snippet 24"
    assert [hit.title for _, hit in selected] == [
        f"Candidate {index}" for index in reversed(range(25))
    ]
    assert trace["candidate_count"] == 25
    assert trace["duration_seconds"] >= 0
    assert action == "fetch_selected"


@pytest.mark.asyncio
async def test_worker_compacts_reused_candidates_without_reducing_inventory():
    class SelectionLLM:
        model_id = "selection-model"

        def __init__(self):
            self.payload = {}

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            self.payload = json.loads(user)
            ids = [item["candidate_id"] for item in self.payload["candidates"]]
            return {
                "selected_candidate_ids": ids,
                "next_action": "fetch_selected",
                "rationale": "keep all candidate identities visible",
            }, LLMResponse("{}", input_tokens=100, output_tokens=20)

    llm = SelectionLLM()
    worker = ResearchWorker(llm, SkillRegistry())
    task = ResearchTask(
        task_id="task-reused-selection",
        objective="Choose reused sources",
        research_dimension="selection subject",
        queries=["selection subject"],
        budget=TaskBudget(max_searches=1, max_fetches=10, max_wall_time_seconds=30),
    )
    candidates = [
        (
            ["query-1"],
            SearchHit(
                title="Reused source",
                url="https://source.example/research",
                snippet="A long source snippet that should not be repeated",
                provider="test",
                rank=1,
            ),
        )
    ]

    selected, trace, _, action = await worker._select_candidates(
        {"run_id": "run-reused-selection", "scope": {}, "claims": [], "unresolved_gaps": []},
        task,
        candidates,
        profile_id="general_research",
        selection_no=1,
        remaining_fetches=10,
        query_texts={"query-1": "selection subject"},
        compact_candidate_urls={"https://source.example/research"},
    )

    assert len(llm.payload["candidates"]) == 1
    assert llm.payload["candidates"][0]["reused_candidate"] is True
    assert "snippet" not in llm.payload["candidates"][0]
    assert trace["candidate_count"] == 1
    assert trace["candidates"][0]["snippet"] == "A long source snippet that should not be repeated"
    assert trace["candidate_payload_compacted_count"] == 1
    assert len(selected) == 1
    assert action == "fetch_selected"


@pytest.mark.asyncio
async def test_worker_randomly_truncates_more_than_fifty_candidates():
    class BatchedSelectionLLM:
        model_id = "batched-selection-model"

        def __init__(self):
            self.payloads = []

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            payload = json.loads(user)
            self.payloads.append(payload)
            candidate_ids = [item["candidate_id"] for item in payload["candidates"]]
            return {
                "selected_candidate_ids": candidate_ids,
                "next_action": "fetch_selected",
                "rationale": "review every supplied candidate",
            }, LLMResponse("{}", input_tokens=len(candidate_ids), output_tokens=5)

    llm = BatchedSelectionLLM()
    worker = ResearchWorker(llm, SkillRegistry())
    task = ResearchTask(
        task_id="task-batched-selection",
        objective="Choose useful sources from a large pool",
        research_dimension="source selection",
        research_object="large selection subject",
        queries=["large selection subject"],
        budget=TaskBudget(max_searches=2, max_fetches=100, max_wall_time_seconds=30),
    )
    candidates = [
        (
            [f"query-{index}"],
            SearchHit(
                title=f"Candidate {index}",
                url=f"https://source-{index}.example/research",
                snippet=f"Complete snippet {index}",
                provider="test",
                rank=index,
            ),
        )
        for index in range(83)
    ]

    selected, trace, response, action = await worker._select_candidates(
        {
            "run_id": "run-batched-selection",
            "scope": {},
            "claims": [],
            "unresolved_gaps": [],
        },
        task,
        candidates,
        profile_id="general_research",
        selection_no=1,
        remaining_fetches=100,
        query_texts={f"query-{index}": f"query text {index}" for index in range(83)},
    )

    assert len(llm.payloads) == 3
    assert [payload["selection_stage"] for payload in llm.payloads] == [
        "batch_review",
        "batch_review",
        "global_merge",
    ]
    assert [len(payload["candidates"]) for payload in llm.payloads] == [50, 33, 83]
    assert llm.payloads[0]["candidate_universe_count"] == 83
    assert llm.payloads[0]["randomly_truncated_count"] == 0
    assert len(selected) == 83
    assert trace["selection_mode"] == "worker_llm_batched"
    assert trace["candidate_count"] == 83
    assert trace["candidate_universe_count"] == 83
    assert trace["randomly_truncated_count"] == 0
    assert not trace["randomly_truncated_candidate_ids"]
    assert trace["batch_count"] == 2
    assert trace["merge_candidate_count"] == 83
    assert not trace["deferred_candidate_ids"]
    assert response is not None
    assert response.input_tokens == 166
    assert action == "fetch_selected"


@pytest.mark.asyncio
async def test_worker_selection_timeout_queues_only_one_fetch_batch():
    class TimeoutSelectionLLM:
        model_id = "timeout-selection-model"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            raise TimeoutError("selector timed out")

    worker = ResearchWorker(TimeoutSelectionLLM(), SkillRegistry())
    task = ResearchTask(
        task_id="task-timeout-selection",
        objective="Choose useful sources",
        research_dimension="source selection",
        queries=["selection subject"],
        budget=TaskBudget(max_searches=2, max_fetches=30, max_wall_time_seconds=30),
    )
    candidates = [
        (
            ["query-1"],
            SearchHit(
                title=f"Candidate {index}",
                url=f"https://timeout-{index}.example/research",
                provider="test",
                rank=index,
            ),
        )
        for index in range(25)
    ]

    selected, trace, response, action = await worker._select_candidates(
        {
            "run_id": "run-timeout-selection",
            "scope": {},
            "claims": [],
            "unresolved_gaps": [],
        },
        task,
        candidates,
        profile_id="general_research",
        selection_no=1,
        remaining_fetches=30,
        query_texts={"query-1": "selection subject"},
    )

    assert [hit.title for _, hit in selected] == ["Candidate 0", "Candidate 1"]
    assert trace["selection_mode"] == "deterministic_small_batch_fallback"
    assert len(trace["deferred_candidate_ids"]) == 23
    assert "TimeoutError" in trace["error"]
    assert response is None
    assert action == "fetch_selected"


@pytest.mark.asyncio
async def test_worker_consumes_selected_queue_before_searching_again():
    class QueueLLM:
        model_id = "queue-model"

        def __init__(self):
            self.selection_calls = 0
            self.active_extractions = 0
            self.peak_extractions = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            payload = json.loads(user)
            if payload.get("phase") == "strategy":
                return {
                    "research_assessment": "consume the selected source queue",
                    "queries": ["queue evidence"],
                    "source_targets": ["fixture sources"],
                    "risks": [],
                }, LLMResponse("{}", input_tokens=5, output_tokens=5)
            if payload.get("phase") == "candidate_selection":
                self.selection_calls += 1
                return {
                    "selected_candidate_ids": [
                        item["candidate_id"] for item in payload["candidates"]
                    ],
                    "next_action": "fetch_selected",
                    "rationale": "all six sources are useful",
                }, LLMResponse("{}", input_tokens=10, output_tokens=5)
            if payload.get("phase") == "reflection":
                return {
                    "result_brief": "four independent sources were verified",
                    "knowledge_gaps": [],
                    "conflicts": [],
                }, LLMResponse("{}", input_tokens=5, output_tokens=5)

            source_uri = payload["source_uri"]
            marker = source_uri.split("//", 1)[-1].split(".", 1)[0]
            quote = f"The verified value for {marker} is {marker}."
            self.active_extractions += 1
            self.peak_extractions = max(
                self.peak_extractions, self.active_extractions
            )
            try:
                await asyncio.sleep(0.02)
                return {
                    "claims": [
                        {
                            "text": quote,
                            "quote": quote,
                            "impact": "medium",
                            "claim_type": "factual",
                            "requirement_ids": ["req-queue"],
                        }
                    ],
                    "knowledge_gaps": [],
                }, LLMResponse("{}", input_tokens=10, output_tokens=5)
            finally:
                self.active_extractions -= 1

    async def search_handler(query: str, max_results: int = 5):
        suffixes = ["com", "net", "org", "edu", "io", "co"]
        return [
            SearchHit(
                title=f"Source {index}",
                url=f"https://source{index}.example.{suffix}",
                snippet=f"Candidate {index}",
                provider="fixture",
                rank=index,
            )
            for index, suffix in enumerate(suffixes, start=1)
        ]

    fetch_calls: list[str] = []

    async def fetch_handler(url: str):
        fetch_calls.append(url)
        marker = url.split("//", 1)[-1].split(".", 1)[0]
        return FetchedDocument(
            requested_url=url,
            final_url=url,
            title=marker,
            content=f"The verified value for {marker} is {marker}.",
            content_type="text/plain",
            http_status=200,
        )

    registry = SkillRegistry()
    registry.register(
        SkillDefinition(
            skill_id="web-search",
            description="fixture search",
            instructions="",
            roles=frozenset({"worker"}),
            profiles=frozenset(),
            handler=search_handler,
            tags=frozenset({"search"}),
            result_kind="search_hits",
        )
    )
    registry.register(
        SkillDefinition(
            skill_id="safe-fetch",
            description="fixture fetch",
            instructions="",
            roles=frozenset({"worker"}),
            profiles=frozenset(),
            handler=fetch_handler,
            tags=frozenset({"fetch"}),
            result_kind="document",
        )
    )

    llm = QueueLLM()
    worker = ResearchWorker(llm, registry)
    task = ResearchTask(
        task_id="task-durable-queue",
        objective="Verify four independent sources",
        research_dimension="queue behavior",
        queries=["queue evidence"],
        requirement_ids=["req-queue"],
        produces_requirement_ids=["req-queue"],
        budget=TaskBudget(max_searches=4, max_fetches=6, max_wall_time_seconds=30),
        sufficiency={
            "min_queries": 1,
            "min_query_coverage": 1,
            "min_successful_extractions": 4,
            "min_effective_sources": 4,
            "min_independent_domains": 4,
            "min_evidence": 4,
        },
    )
    state = {
        "run_id": "run-durable-queue",
        "request": {"profile_id": "general_research", "constraints": {}},
        "scope": {
            "current_date": "2026-08-06",
            "date_boundary": None,
            "source_urls": [],
            "options": {"max_worker_iterations": 4},
        },
        "plan_history": [
            {
                "requirements": [
                    {
                        "requirement_id": "req-queue",
                        "text": "Verify the queue sources",
                        "required_elements": ["source value"],
                    }
                ]
            }
        ],
        "sources": [],
        "evidence": [],
        "claims": [],
        "links": [],
        "search_traces": [],
        "fetch_traces": [],
        "usage_events": [],
        "unresolved_gaps": [],
        "worker_memory": {},
        "dependency_context": {},
        "started_at": time.time(),
    }

    result = await worker.run(state, task)

    assert llm.selection_calls == 1
    assert llm.peak_extractions == 2
    # The configured eight-URL batch starts this entire six-item selected
    # queue concurrently. Sufficiency still prevents another search window.
    assert len(fetch_calls) == 6
    assert result["contributions"][0]["sufficiency_passed"] is True


@pytest.mark.asyncio
async def test_global_extraction_pool_caps_concurrency_at_twelve():
    class ExtractionLLM:
        model_id = "extraction-pool-model"

        def __init__(self):
            self.active = 0
            self.peak = 0
            self.twelve_started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            payload = json.loads(user)
            quote = "The verified release value is 12."
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.peak == 12:
                self.twelve_started.set()
            try:
                await self.release.wait()
                return {
                    "claims": [
                        {
                            "text": quote,
                            "quote": quote,
                            "impact": "medium",
                            "claim_type": "factual",
                            "requirement_ids": ["req-pool"],
                        }
                    ],
                    "knowledge_gaps": [],
                }, LLMResponse("{}", input_tokens=10, output_tokens=5)
            finally:
                self.active -= 1

    llm = ExtractionLLM()
    worker = ResearchWorker(llm, SkillRegistry())
    task = ResearchTask(
        task_id="task-extraction-pool",
        objective="Verify the extraction pool",
        research_dimension="pool concurrency",
        queries=["pool evidence"],
        requirement_ids=["req-pool"],
        budget=TaskBudget(max_searches=1, max_fetches=13, max_wall_time_seconds=30),
    )
    state = {
        "run_id": "run-extraction-pool",
        "request": {
            "mode": "standard",
            "profile_id": "general_research",
            "constraints": {},
        },
        "scope": {},
        "started_at": time.time(),
        "search_schemas": [],
    }
    outcomes = []
    for index in range(13):
        url = f"https://source{index}.example.com/release"
        outcomes.append(
            {
                "hit": SearchHit(
                    title=f"Source {index}",
                    url=url,
                    provider="fixture",
                    rank=index,
                ),
                "document": FetchedDocument(
                    requested_url=url,
                    final_url=url,
                    title=f"Source {index}",
                    content="The verified release value is 12.",
                    content_type="text/plain",
                    http_status=200,
                ),
                "fetch_operation": f"fetch-{index}",
            }
        )

    extraction_tasks = [
        asyncio.create_task(
            worker._extract_fetched_document(
                state=state,
                task=task,
                outcome=outcome,
                requirement_details=[],
                active_search_schema=None,
                progress_fields={},
            )
        )
        for outcome in outcomes
    ]
    await asyncio.wait_for(llm.twelve_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert llm.active == 12
    assert not extraction_tasks[-1].done()

    llm.release.set()
    results = await asyncio.gather(*extraction_tasks)

    assert llm.peak == 12
    assert len(results) == 13


@pytest.mark.asyncio
async def test_verified_claim_is_partial_until_producer_passes_sufficiency():
    state = {
        "run_id": "run-requirement-status",
        "round_no": 1,
        "wave_no": 1,
        "wave_history": [{"wave_id": "r1-w1"}],
        "current_plan": {
            "requirements": [{"requirement_id": "req-table"}]
        },
        "current_tasks": [
            {
                "task_id": "task-table",
                "requirement_ids": ["req-table"],
                "produces_requirement_ids": ["req-table"],
            }
        ],
        "task_runtime": [
            {"task_id": "task-table", "status": "pending"}
        ],
        "contributions": [],
        "sources": [
            {
                "source_id": "source-1",
                "canonical_uri": "https://example.org/source",
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_id": "source-1",
                "requirement_ids": ["req-table"],
            }
        ],
        "claims": [
            {"claim_id": "claim-1", "requirement_ids": ["req-table"]}
        ],
        "links": [
            {
                "link_id": "link-1",
                "claim_id": "claim-1",
                "evidence_id": "evidence-1",
                "relation": "supports",
                "entailment_status": "verified",
            }
        ],
    }

    partial = await ResearchGraph._deterministic_checks(None, state)
    partial_status = partial["requirement_status"][0]
    assert partial_status["status"] == "partial"
    assert partial_status["completed_sufficient_producer_task_ids"] == []

    state["task_runtime"] = [{"task_id": "task-table", "status": "completed"}]
    state["contributions"] = [
        {
            "task_id": "task-table",
            "status": "completed",
            "usable": True,
            "sufficiency_passed": True,
        }
    ]
    sufficient = await ResearchGraph._deterministic_checks(None, state)
    sufficient_status = sufficient["requirement_status"][0]
    assert sufficient_status["status"] == "sufficient"
    assert sufficient_status["completed_sufficient_producer_task_ids"] == [
        "task-table"
    ]


@pytest.mark.asyncio
async def test_composite_requirement_coverage_is_advisory_and_element_aware():
    state = {
        "run_id": "run-composite-requirement-status",
        "round_no": 1,
        "wave_no": 1,
        "wave_history": [{"wave_id": "r1-w1"}],
        "current_plan": {
            "requirements": [
                {
                    "requirement_id": "req-composite",
                    "text": "Explain the legal regime.",
                    "required_elements": [
                        "legal basis",
                        "liability type",
                        "defenses",
                    ],
                }
            ]
        },
        "current_tasks": [
            {
                "task_id": "task-composite",
                "requirement_ids": ["req-composite"],
                "produces_requirement_ids": ["req-composite"],
            }
        ],
        "task_runtime": [{"task_id": "task-composite", "status": "completed"}],
        "contributions": [
            {
                "task_id": "task-composite",
                "status": "completed",
                "usable": True,
                "sufficiency_passed": True,
            }
        ],
        "sources": [
            {
                "source_id": "source-composite",
                "canonical_uri": "https://example.org/source",
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-composite",
                "source_id": "source-composite",
                "requirement_ids": ["req-composite"],
            }
        ],
        "claims": [
            {
                "claim_id": "claim-composite",
                "requirement_ids": ["req-composite"],
            }
        ],
        "links": [
            {
                "link_id": "link-composite",
                "claim_id": "claim-composite",
                "evidence_id": "evidence-composite",
                "relation": "supports",
                "entailment_status": "verified",
            }
        ],
    }

    result = await ResearchGraph._deterministic_checks(None, state)
    status = result["requirement_status"][0]

    assert status["status"] == "partial"
    assert status["operational_status"] == "sufficient"
    assert status["semantic_status"] == "unassessed"
    assert status["coverage_authority"] == "advisory"
    assert status["semantic_completeness_assessed"] is False
    assert status["required_elements"] == [
        "legal basis",
        "liability type",
        "defenses",
    ]
    assert [item["status"] for item in status["required_element_statuses"]] == [
        "unassessed",
        "unassessed",
        "unassessed",
    ]


@pytest.mark.asyncio
async def test_requirement_stays_partial_until_all_complementary_producers_pass():
    state = {
        "run_id": "run-complementary-producers",
        "round_no": 1,
        "wave_no": 1,
        "wave_history": [{"wave_id": "r1-w1"}],
        "current_plan": {
            "requirements": [{"requirement_id": "req-table"}]
        },
        "current_tasks": [
            {
                "task_id": "task-early-period",
                "requirement_ids": ["req-table"],
                "produces_requirement_ids": ["req-table"],
            },
            {
                "task_id": "task-late-period",
                "requirement_ids": ["req-table"],
                "produces_requirement_ids": ["req-table"],
            },
        ],
        "task_runtime": [
            {"task_id": "task-early-period", "status": "completed"},
            {"task_id": "task-late-period", "status": "pending"},
        ],
        "contributions": [
            {
                "task_id": "task-early-period",
                "status": "completed",
                "usable": True,
                "sufficiency_passed": True,
            }
        ],
        "sources": [
            {
                "source_id": "source-1",
                "canonical_uri": "https://example.org/source",
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-1",
                "source_id": "source-1",
                "requirement_ids": ["req-table"],
            }
        ],
        "claims": [
            {"claim_id": "claim-1", "requirement_ids": ["req-table"]}
        ],
        "links": [
            {
                "link_id": "link-1",
                "claim_id": "claim-1",
                "evidence_id": "evidence-1",
                "relation": "supports",
                "entailment_status": "verified",
            }
        ],
    }

    result = await ResearchGraph._deterministic_checks(None, state)
    status = result["requirement_status"][0]
    assert status["status"] == "partial"
    assert status["all_producers_passed_sufficiency"] is False
    assert status["completed_sufficient_producer_task_ids"] == [
        "task-early-period"
    ]


@pytest.mark.asyncio
async def test_collection_cell_coverage_does_not_regress_with_task_sufficiency():
    """Evidence coverage is stable even while a new producer is pending."""

    state = {
        "run_id": "run-cell-coverage-separation",
        "round_no": 2,
        "wave_no": 1,
        "wave_history": [{"wave_id": "r2-w1"}],
        "current_plan": {
            "requirements": [
                {
                    "requirement_id": "req-cell",
                    "item_key": "beijing",
                    "field_key": "operating_mileage",
                }
            ],
            "coverage_blueprint": {
                "repeated_case_fields": ["operating_mileage"],
                "candidate_cases": [
                    {
                        "case_id": "beijing",
                        "fields": ["operating_mileage"],
                        "field_requirement_ids": {
                            "operating_mileage": "req-cell"
                        },
                    }
                ],
            },
        },
        "current_tasks": [
            {
                "task_id": "task-new-producer",
                "requirement_ids": ["req-cell"],
                "produces_requirement_ids": ["req-cell"],
            }
        ],
        "task_runtime": [{"task_id": "task-new-producer", "status": "pending"}],
        "contributions": [],
        "sources": [
            {
                "source_id": "source-cell",
                "canonical_uri": "https://example.org/cell",
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-cell",
                "source_id": "source-cell",
                "requirement_ids": ["req-cell"],
            }
        ],
        "claims": [{"claim_id": "claim-cell", "requirement_ids": ["req-cell"]}],
        "links": [
            {
                "link_id": "link-cell",
                "claim_id": "claim-cell",
                "evidence_id": "evidence-cell",
                "relation": "supports",
                "entailment_status": "verified",
            }
        ],
    }

    result = await ResearchGraph._deterministic_checks(None, state)

    assert result["requirement_status"][0]["status"] == "partial"
    assert result["requirement_status"][0]["evidence_coverage_status"] == "linked"
    assert result["collection_coverage"]["covered_cell_count"] == 1
    assert result["collection_coverage"]["cell_coverage_ratio"] == 1.0


def test_prior_candidates_do_not_leak_from_unrelated_indexed_worker():
    current = ResearchTask(
        task_id="task-current",
        objective="Research Husanovic",
        research_dimension="Husanovic",
        research_object="Jasmina Husanovic",
        queries=[],
        requirement_ids=[],
        budget=TaskBudget(max_searches=0, max_fetches=4, max_wall_time_seconds=30),
    )
    unrelated = ResearchTask(
        task_id="task-boris",
        objective="Research Boris Takk",
        research_dimension="Boris Takk",
        research_object="Boris Takk",
        queries=["Boris Takk"],
        requirement_ids=["req-001"],
        budget=TaskBudget(max_searches=2, max_fetches=4, max_wall_time_seconds=30),
    )
    state = {
        "tasks": [unrelated.model_dump(mode="json"), current.model_dump(mode="json")],
        "current_tasks": [current.model_dump(mode="json")],
        "fetch_traces": [],
        "search_traces": [
            {
                "trace_id": "search-boris",
                "run_id": "run-isolation",
                "task_id": unrelated.task_id,
                "attempt": 1,
                "query_id": "query-boris",
                "query": "Boris Takk",
                "requirement_ids": ["req-001"],
                "results": [
                    {
                        "title": "Boris source",
                        "url": "https://example.org/boris",
                        "provider": "test",
                    }
                ],
            }
        ],
    }

    assert _prior_search_candidate_groups(state, current) == []


def test_focused_replan_reserves_less_than_the_completed_broad_round():
    required = _required_next_round_seconds(
        round_elapsed_seconds=1_400,
        soft_wall_time_seconds=2_040,
        max_rounds=4,
    )

    assert required == 255
    assert 300 >= required

def _fetch_only_task() -> ResearchTask:
    return ResearchTask(
        task_id="task-recovery",
        objective="Recover remaining evidence",
        research_dimension="missing evidence",
        round_no=3,
        queries=["unused query"],
        requirement_ids=["req-002"],
        budget=TaskBudget(
            max_searches=0,
            max_fetches=4,
            max_wall_time_seconds=30,
        ),
    )


def test_fetch_only_round_reuses_relevant_unattempted_search_hits():
    state = {
        "search_traces": [
            {
                "trace_id": "search-1",
                "run_id": "run-1",
                "task_id": "task-old",
                "attempt": 1,
                "query_id": "query-1",
                "query": "relevant query",
                "requirement_ids": ["req-002"],
                "results": [
                    {
                        "title": "Already attempted",
                        "url": "https://example.com/attempted",
                        "provider": "test",
                    },
                    {
                        "title": "Remaining candidate",
                        "url": "https://example.org/remaining",
                        "provider": "test",
                    },
                ],
            },
            {
                "trace_id": "search-2",
                "run_id": "run-1",
                "task_id": "task-other",
                "attempt": 1,
                "query_id": "query-2",
                "query": "unrelated query",
                "requirement_ids": ["req-009"],
                "results": [
                    {
                        "title": "Wrong requirement",
                        "url": "https://example.net/unrelated",
                        "provider": "test",
                    }
                ],
            },
        ],
        "fetch_traces": [{"url": "https://example.com/attempted"}],
    }

    groups = _prior_search_candidate_groups(state, _fetch_only_task())

    assert [[hit.url for _, hit in group] for group in groups] == [
        ["https://example.org/remaining"]
    ]
    assert _has_unattempted_search_candidates(state)


def test_fetch_only_round_stops_when_all_candidates_were_attempted():
    state = {
        "search_traces": [
            {
                "results": [
                    {"url": "https://example.com/attempted"},
                ]
            }
        ],
        "fetch_traces": [{"url": "https://example.com/attempted"}],
    }

    assert not _has_unattempted_search_candidates(state)
