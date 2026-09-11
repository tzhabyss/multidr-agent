from deepresearch.budget import (
    CachedFetchFailure,
    SharedDomainCircuitBreaker,
    SharedFetchBudgetPool,
    SharedFetchCoordinator,
    SharedSearchBudgetPool,
    allocate_task_budgets,
    remaining_task_budget,
    split_task_budget,
)
from deepresearch.agents import (
    _extraction_completion_deadline_at,
    _resolve_sufficiency,
)
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import ResearchMode, ResearchRequest, RunBudget, RunStatus
from deepresearch.schemas import ResearchTask, TaskBudget
from deepresearch.service import ResearchService

import pytest
import time
import asyncio

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


def _budget(**updates):
    values = {
        "max_total_tokens": 1000,
        "max_total_searches": 2,
        "max_total_fetches": 2,
        "max_wall_time_seconds": 30,
        "planner_reserved_tokens": 100,
        "synthesizer_reserved_tokens": 200,
    }
    values.update(updates)
    return RunBudget(**values)


def test_extraction_started_before_soft_cutoff_gets_bounded_drain_window():
    config = load_runtime_config()
    state = {
        "request": {"mode": "standard", "profile_id": "drb_evaluation"},
        "started_at": 900.0,
        "worker_tool_deadline_at": 1010.0,
        "worker_reflection_deadline_at": 1120.0,
        "run_budget": {"max_wall_time_seconds": 4080},
    }

    deadline = _extraction_completion_deadline_at(state, config)

    assert deadline == 1100.0


def test_default_budget_keeps_research_capacity_with_realistic_lead_reserve():
    budget = RunBudget.default()

    assert "max_total_tokens" not in budget.model_dump()
    assert "planner_reserved_tokens" not in budget.model_dump()
    assert "synthesizer_reserved_tokens" not in budget.model_dump()


def test_source_quality_tier_is_runtime_policy_not_planner_output():
    assert _resolve_sufficiency({"max_source_quality_tier": 1}).max_source_quality_tier == 3
    assert (
        _resolve_sufficiency(
            {"max_source_quality_tier": 1},
            {"max_source_quality_tier": 2},
        ).max_source_quality_tier
        == 2
    )


def test_parallel_allocations_never_exceed_global_budget():
    budget = _budget(max_total_searches=1, max_total_fetches=2)
    tasks = allocate_task_budgets(budget, 3)
    assert sum(item.max_searches for item in tasks) <= 1
    assert sum(item.max_fetches for item in tasks) <= 2


def test_round_pool_is_split_without_exceeding_available_budget():
    pool = TaskBudget(max_searches=5, max_fetches=8, max_wall_time_seconds=30)
    allocations = split_task_budget(pool, 3)
    assert sum(item.max_searches for item in allocations) == 5
    assert sum(item.max_fetches for item in allocations) == 8


@pytest.mark.asyncio
async def test_shared_fetch_pool_reclaims_unused_worker_reservation():
    pool = SharedFetchBudgetPool()
    await pool.configure_round("run-shared", 1, {"worker-a": 1, "worker-b": 2})

    assert await pool.acquire("run-shared", 1, "worker-a") is True
    assert await pool.acquire("run-shared", 1, "worker-b") is True
    assert await pool.complete("run-shared", 1, "worker-b") == 1
    assert await pool.acquire("run-shared", 1, "worker-a") is True
    assert await pool.acquire("run-shared", 1, "worker-a") is False

    snapshot = await pool.snapshot("run-shared", 1)
    assert snapshot is not None
    assert snapshot["total"] == 3
    assert snapshot["consumed"] == 3
    assert snapshot["reclaimed"] == 0


@pytest.mark.asyncio
async def test_unconfigured_shared_fetch_pool_preserves_standalone_worker_fallback():
    pool = SharedFetchBudgetPool()

    assert await pool.acquire("missing", 1, "worker") is None


@pytest.mark.asyncio
async def test_shared_search_pool_atomically_caps_logical_queries():
    pool = SharedSearchBudgetPool()
    await pool.configure_round("run-search", 1, {"worker-a": 3, "worker-b": 2})

    assert await pool.acquire("run-search", 1, "worker-a") is True
    assert await pool.acquire("run-search", 1, "worker-b") is True
    assert await pool.acquire("run-search", 1, "worker-a", units=3) is True
    assert await pool.acquire("run-search", 1, "worker-a") is False

    snapshot = await pool.snapshot("run-search", 1)
    assert snapshot is not None
    assert snapshot["total"] == 5
    assert snapshot["consumed"] == 5
    assert sum(snapshot["reservations"].values()) + snapshot["reclaimed"] == 0


@pytest.mark.asyncio
async def test_url_owner_is_exclusive_and_failed_owner_can_release():
    coordinator = SharedFetchCoordinator()

    assert await coordinator.claim_url("run-owner", "https://example.com/a", "a")
    assert not await coordinator.claim_url(
        "run-owner", "https://example.com/a", "b"
    )
    assert not await coordinator.release_url(
        "run-owner", "https://example.com/a", "b"
    )
    assert await coordinator.release_url(
        "run-owner", "https://example.com/a", "a"
    )
    assert await coordinator.claim_url("run-owner", "https://example.com/a", "b")


@pytest.mark.asyncio
async def test_shared_fetch_queue_caps_global_concurrency_at_twelve():
    coordinator = SharedFetchCoordinator(max_concurrency=12)
    active = 0
    peak = 0
    release = asyncio.Event()

    async def operation(index: int):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if peak >= 12:
            release.set()
        await release.wait()
        await asyncio.sleep(0)
        active -= 1
        return {"content": f"document-{index}"}

    tasks = [
        asyncio.create_task(
            coordinator.fetch(
                "run-queue",
                f"https://example.com/{index}",
                lambda index=index: operation(index),
            )
        )
        for index in range(20)
    ]
    await asyncio.wait_for(release.wait(), timeout=1)
    await asyncio.gather(*tasks)

    snapshot = await coordinator.snapshot("run-queue")
    assert peak == 12
    assert snapshot["peak"] == 12
    assert snapshot["max_concurrency"] == 12


@pytest.mark.asyncio
async def test_shared_fetch_queue_caps_one_domain_without_reducing_global_limit():
    coordinator = SharedFetchCoordinator(
        max_concurrency=8,
        max_concurrency_per_domain=4,
    )
    active = 0
    peak = 0
    release = asyncio.Event()

    async def operation(index: int):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if peak >= 4:
            release.set()
        await release.wait()
        await asyncio.sleep(0)
        active -= 1
        return {"content": f"document-{index}"}

    tasks = [
        asyncio.create_task(
            coordinator.fetch(
                "run-domain-limit",
                f"https://example.com/{index}",
                lambda index=index: operation(index),
                domain="example.com",
            )
        )
        for index in range(10)
    ]
    await asyncio.wait_for(release.wait(), timeout=1)
    await asyncio.gather(*tasks)

    assert peak == 4


@pytest.mark.asyncio
async def test_shared_fetch_coordinator_negative_caches_failed_url_across_workers():
    coordinator = SharedFetchCoordinator(failure_ttl_seconds=60)
    calls = 0

    async def fail_once():
        nonlocal calls
        calls += 1
        raise TimeoutError("upstream timed out")

    with pytest.raises(TimeoutError):
        await coordinator.fetch("shared-run", "https://broken.example/a", fail_once)
    with pytest.raises(CachedFetchFailure, match="negative_cache:TimeoutError"):
        await coordinator.fetch("shared-run", "https://broken.example/a", fail_once)

    assert calls == 1

    await coordinator.clear_run("shared-run")
    with pytest.raises(TimeoutError):
        await coordinator.fetch("shared-run", "https://broken.example/a", fail_once)
    assert calls == 2


@pytest.mark.asyncio
async def test_shared_fetch_pool_can_borrow_another_live_worker_reservation():
    pool = SharedFetchBudgetPool()
    await pool.configure_round("run-borrow", 1, {"worker-a": 0, "worker-b": 2})

    assert await pool.acquire("run-borrow", 1, "worker-a") is True
    snapshot = await pool.snapshot("run-borrow", 1)

    assert snapshot is not None
    assert snapshot["consumed"] == 1
    assert snapshot["reservations"] == {"worker-a": 0, "worker-b": 1}


@pytest.mark.asyncio
async def test_domain_circuit_opens_after_consecutive_failures_and_resets_on_success():
    circuit = SharedDomainCircuitBreaker(failure_threshold=2)

    assert await circuit.can_attempt("run-domain", "example.com")
    assert not await circuit.record_failure("run-domain", "example.com")
    await circuit.record_success("run-domain", "example.com")
    assert not await circuit.record_failure("run-domain", "example.com")
    assert await circuit.record_failure("run-domain", "example.com")
    assert not await circuit.can_attempt("run-domain", "example.com")
    assert await circuit.can_attempt("run-domain", "other.example")

    assert await circuit.snapshot("run-domain") == {
        "consecutive_failures": {"example.com": 2},
        "open_domains": ["example.com"],
    }


@pytest.mark.asyncio
async def test_domain_circuit_half_opens_after_cooldown(monkeypatch):
    clock = 100.0
    monkeypatch.setattr("deepresearch.budget.time.monotonic", lambda: clock)
    circuit = SharedDomainCircuitBreaker(
        failure_threshold=2, cooldown_seconds=30
    )

    await circuit.record_failure("run-cooldown", "example.com")
    await circuit.record_failure("run-cooldown", "example.com")
    assert not await circuit.can_attempt("run-cooldown", "example.com")

    clock = 131.0
    assert await circuit.can_attempt("run-cooldown", "example.com")
    assert await circuit.record_failure("run-cooldown", "example.com")


@pytest.mark.asyncio
async def test_fetch_coordinator_coalesces_in_flight_and_bounds_completed_cache():
    coordinator = SharedFetchCoordinator(
        max_entries_per_run=2,
        max_content_chars_per_run=20,
        max_cached_document_chars=10,
    )
    calls = 0

    async def fetch_once():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return {"content": "shared"}

    first, second = await asyncio.gather(
        coordinator.fetch("run-cache", "https://example.com/a", fetch_once),
        coordinator.fetch("run-cache", "https://example.com/a", fetch_once),
    )
    cached = await coordinator.lookup("run-cache", "https://example.com/a")

    assert calls == 1
    assert {first[1], second[1]} == {"fetched", "shared"}
    assert cached == ({"content": "shared"}, "cached")
    assert await coordinator.snapshot("run-cache") == {
        "completed": 1,
        "in_flight": 0,
        "content_chars": 6,
        "active": 0,
        "pending": 0,
        "peak": 1,
        "max_concurrency": 16,
    }

    await coordinator.fetch(
        "run-cache",
        "https://example.com/too-large",
        lambda: asyncio.sleep(0, result={"content": "x" * 11}),
    )
    assert await coordinator.lookup("run-cache", "https://example.com/too-large") is None


def test_replan_receives_only_remaining_consumed_budget():
    budget = _budget()
    events = [
        {
            "usage_event_id": "use-1",
            "input_tokens": 100,
            "output_tokens": 50,
            "searches": 1,
            "fetches": 1,
        }
    ]
    remaining = remaining_task_budget(budget, events, round_no=2, max_rounds=3)
    assert remaining.max_searches == 1
    assert remaining.max_fetches == 1


def test_first_round_exposes_full_global_remainder_without_stranding_reserve():
    budget = _budget(max_total_searches=200, max_total_fetches=336)

    remaining = remaining_task_budget(budget, [], round_no=1, max_rounds=4)

    assert remaining.max_searches == 200
    assert remaining.max_fetches == 336


@pytest.mark.asyncio
async def test_token_usage_is_recorded_but_does_not_stop_the_run(tmp_path):
    budget = _budget(
        max_total_tokens=200,
        max_total_searches=1,
        max_total_fetches=1,
        planner_reserved_tokens=50,
        synthesizer_reserved_tokens=100,
    )
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            ResearchRequest(question="Respect a tiny budget", mode=ResearchMode.DEEP_SEARCH),
            budget=budget,
        )

    assert result.status != RunStatus.FAILED_BUDGET_EXHAUSTED
    assert result.usage["total_tokens"] > 0
    assert result.usage["searches"] <= budget.max_total_searches
    assert result.usage["fetches"] <= budget.max_total_fetches


@pytest.mark.asyncio
async def test_expired_run_wall_budget_prevents_tool_execution(tmp_path):
    search = FakeSearchBackend()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    budget = _budget(max_wall_time_seconds=1)
    task = ResearchTask(
        task_id="expired-task",
        objective="Do not execute",
        research_dimension="timeout",
        queries=["must not run"],
        budget=TaskBudget(
            max_tokens=500,
            max_searches=1,
            max_fetches=1,
            max_wall_time_seconds=1,
        ),
    )
    async with service:
        assert service._graph is not None
        result = await service._graph._worker_node(
            {
                "run_id": "expired-run",
                "request": {"profile_id": "general_research"},
                "scope": {"source_urls": []},
                "run_budget": budget.model_dump(),
                "started_at": time.time() - 2,
                "current_task": task.model_dump(),
                "review_round": 0,
            }
        )

    assert result["contributions"][0]["stop_reason"] == "budget"
    assert search.calls == []


@pytest.mark.asyncio
async def test_task_wall_timeout_returns_partial_audit_trace(tmp_path):
    class SlowSearch(FakeSearchBackend):
        async def search(self, query: str, max_results: int = 5):
            self.calls.append(query)
            await asyncio.sleep(2)
            return []

    search = SlowSearch()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    task = ResearchTask(
        task_id="timeout-audit-task",
        objective="Preserve timeout trace",
        research_dimension="timeout",
        queries=["slow query"],
        budget=TaskBudget(
            max_tokens=1000,
            max_searches=1,
            max_fetches=1,
            max_wall_time_seconds=1,
        ),
    )
    async with service:
        assert service._graph is not None
        result = await service._graph._worker_node(
            {
                "run_id": "timeout-audit-run",
                "request": {
                    "profile_id": "general_research",
                    "mode": "deep_search",
                    "constraints": {},
                },
                "scope": {"source_urls": []},
                "run_budget": _budget(max_wall_time_seconds=10).model_dump(),
                "started_at": time.time(),
                "current_task": task.model_dump(),
                "review_round": 0,
                "usage_events": [],
            }
        )

    assert result["contributions"][0]["stop_reason"] == "timeout"
    assert result["search_traces"][0]["error"] == "task_wall_time_timeout"
    assert sum(item["searches"] for item in result["usage_events"]) == 1
