from __future__ import annotations

from deepresearch.agents import ResearchWorker, _focused_extraction_windows
from deepresearch.failure_memory import active_failures, failures_from_fetch_traces
from deepresearch.frontier import FrontierRegistry
from deepresearch.runtime_config import GraphEngineeringRuntimeConfig
from deepresearch.schemas import ResearchTask, TaskBudget
from deepresearch.search_sensors import evaluate_progress, progress_snapshot
from deepresearch import state as state_ops


def _task(
    task_id: str,
    objective: str,
    *,
    depends_on: list[str] | None = None,
    expected_value: float = 0.5,
    priority: int = 1,
    queries: list[str] | None = None,
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        objective=objective,
        research_dimension="market size",
        depends_on=depends_on or [],
        expected_value=expected_value,
        priority=priority,
        queries=queries or [objective],
        budget=TaskBudget(
            max_searches=3,
            max_fetches=6,
            max_wall_time_seconds=120,
        ),
    )


def test_p0_ablation_mode_and_override_resolution():
    legacy = GraphEngineeringRuntimeConfig(p0_ablation_mode="legacy")
    assert not any(legacy.resolved_p0_features().values())

    mixed = GraphEngineeringRuntimeConfig(
        p0_ablation_mode="enhanced",
        p0_features={"frontier": "legacy"},
    )
    assert mixed.p0_feature_enabled("focused_page_processing")
    assert not mixed.p0_feature_enabled("frontier")
    assert mixed.p0_ablation_snapshot()["context_projection_changed"] is False


def test_enhanced_frontier_merges_duplicates_and_bumps_priority():
    first = _task("t1", "Find annual market size", priority=4, queries=["market size"])
    duplicate = _task(
        "t2",
        "Find annual market size",
        priority=1,
        expected_value=0.9,
        queries=["annual market size official"],
    )
    registry = FrontierRegistry([first, duplicate], [], enhanced=True)

    assert list(registry.tasks) == ["t1"]
    merged = registry.tasks["t1"]
    assert merged.priority == 1
    assert merged.expected_value == 0.9
    assert merged.queries == ["market size", "annual market size official"]
    assert registry.runtime["t1"]["merged_task_ids"] == ["t2"]


def test_enhanced_frontier_blocks_cycles_and_evicts_low_value_pending_tasks():
    cyclic = FrontierRegistry(
        [
            _task("a", "A", depends_on=["b"]),
            _task("b", "B", depends_on=["a"]),
        ],
        [],
        enhanced=True,
    )
    assert {cyclic.runtime[item]["stop_reason"] for item in ("a", "b")} == {
        "frontier_cycle_detected"
    }

    capped = FrontierRegistry(
        [
            _task("high", "High", expected_value=0.9, priority=1),
            _task("low", "Low", expected_value=0.1, priority=9),
        ],
        [],
        enhanced=True,
        max_pending_tasks=1,
    )
    assert capped.runtime["low"]["status"] == "superseded"
    assert capped.runtime["high"]["status"] == "pending"


def test_failure_memory_legacy_and_enhanced_are_distinct():
    traces = [
        {
            "url": "https://example.test/report",
            "provider": "web",
            "status": "failed",
            "error": "ConnectError: connection failed",
        }
    ]
    legacy = failures_from_fetch_traces(
        run_id="run-1", task_id="task-1", traces=traces, enhanced=False
    )
    enhanced = failures_from_fetch_traces(
        run_id="run-1", task_id="task-1", traces=traces, enhanced=True
    )
    assert len(legacy) == 1
    assert legacy[0].scope_type == "task"
    assert len(enhanced) == 1
    assert enhanced[0].scope_type == "url"
    assert set(enhanced[0].metadata["scope_indexes"]) == {
        "task", "url", "domain", "source_family"
    }
    assert all(not item.metadata["semantic_absence_inferred"] for item in enhanced)


def test_enhanced_failure_memory_ignores_unused_prefetch_and_resolves_route_failure():
    failed = failures_from_fetch_traces(
        run_id="run-1",
        task_id="task-1",
        traces=[
            {
                "url": "https://example.test/report",
                "provider": "web",
                "status": "failed",
                "error": "ConnectError: connection failed",
            }
        ],
        enhanced=True,
    )
    assert len(active_failures(failed)) == 1

    resolved = failures_from_fetch_traces(
        run_id="run-1",
        task_id="task-1",
        traces=[
            {
                "url": "https://example.test/report",
                "provider": "web",
                "status": "prefetched_unused",
                "error": "prefetch_not_consumed",
                "content_chars": 12_401,
            }
        ],
        existing=failed,
        enhanced=True,
    )
    assert len(resolved) == 1
    assert resolved[0].failure_id == failed[0].failure_id
    assert resolved[0].metadata["resolved"] is True
    assert not active_failures(resolved)

    clean_prefetch = failures_from_fetch_traces(
        run_id="run-1",
        task_id="task-1",
        traces=[
            {
                "url": "https://example.test/other",
                "status": "prefetched_unused",
                "error": "prefetch_not_consumed",
                "content_chars": 8_000,
            }
        ],
        enhanced=True,
    )
    assert clean_prefetch == []


def test_worker_failure_guidance_reads_enhanced_types_and_skips_resolved_records():
    failed = failures_from_fetch_traces(
        run_id="run-1",
        task_id="task-1",
        traces=[
            {
                "url": "https://example.test/report",
                "provider": "web",
                "status": "failed",
                "error": "ConnectError: connection failed",
            }
        ],
        enhanced=True,
    )
    task = _task("task-1", "Find the report")
    rows = ResearchWorker._recent_fetch_failures(
        {"fetch_traces": [], "failure_memory": [item.model_dump() for item in failed]},
        task,
    )
    assert len(rows) == 1
    assert rows[0]["error_class"] == "ConnectError"
    assert rows[0]["retry_same_route"] is False

    resolved = failures_from_fetch_traces(
        run_id="run-1",
        task_id="task-1",
        traces=[
            {
                "url": "https://example.test/report",
                "status": "fetched",
                "content_chars": 4_000,
                "accepted_claims": 1,
            }
        ],
        existing=failed,
        enhanced=True,
    )
    assert ResearchWorker._recent_fetch_failures(
        {"fetch_traces": [], "failure_memory": [item.model_dump() for item in resolved]},
        task,
    ) == []


def test_enhanced_sensor_counts_unique_active_failures_per_source_family():
    failures = failures_from_fetch_traces(
        run_id="run-1",
        task_id="task-1",
        traces=[
            {
                "url": "https://example.test/a",
                "provider": "web",
                "status": "failed",
                "error": "timeout",
            },
            {
                "url": "https://example.test/b",
                "provider": "web",
                "status": "failed",
                "error": "timeout",
            },
        ],
        enhanced=True,
    )
    snapshot = progress_snapshot(
        {
            "search_traces": [],
            "fetch_traces": [],
            "evidence_bindings": [],
            "entity_candidates": [],
            "sources": [],
            "cell_coverage": {},
            "failure_memory": [item.model_dump() for item in failures],
        },
        window_id="w1",
        enhanced=True,
    )
    assert snapshot["repeated_source_family_failure_count"] == 1


def test_append_only_reducer_preserves_semantic_id_collision():
    original = {
        "evidence_id": "ev-1",
        "source_id": "source-1",
        "quote": "Value is 10.",
        "quote_hash": "q1",
        "location": {},
        "requirement_ids": [],
    }
    conflicting = {**original, "quote": "Value is 12.", "quote_hash": "q2"}
    legacy = state_ops.merge_evidence([original], [conflicting])
    enhanced = state_ops.merge_evidence_append_only([original], [conflicting])

    assert legacy == [conflicting]
    assert len(enhanced) == 2
    assert enhanced[0] == original
    assert {item["quote"] for item in enhanced} == {"Value is 10.", "Value is 12."}


def test_focused_page_processing_preserves_offsets_and_has_safe_no_match():
    task = _task("focus", "Find Tokyo precipitation probability")
    task = task.model_copy(
        update={
            "target_field_keys": ["precipitation_probability"],
            "key_questions": ["What is Tokyo's precipitation probability?"],
        }
    )
    prefix = "navigation text " * 800
    content = prefix + "Tokyo precipitation_probability is 70 percent." + " footer" * 800
    windows, terms = _focused_extraction_windows(content, task)

    assert windows
    assert terms
    start, end, chunk, _ = windows[0]
    assert content[start:end] == chunk
    assert "70 percent" in chunk

    no_match, no_terms = _focused_extraction_windows(
        "unrelated short page", task
    )
    assert no_match == []
    assert no_terms == []


def test_enhanced_sensor_advises_before_it_drains():
    state = {
        "run_id": "run-sensor",
        "search_traces": [
            {
                "query_id": "q1",
                "query": "official Tokyo rainfall",
                "results": [{"url": "https://example.test/a"}],
            },
            {
                "query_id": "q2",
                "query": "rainfall Tokyo official",
                "results": [{"url": "https://example.test/b"}],
            },
        ],
        "fetch_traces": [],
        "evidence_bindings": [],
        "entity_candidates": [],
        "sources": [],
        "cell_coverage": {},
    }
    first = progress_snapshot(state, window_id="w1", enhanced=True)
    second = progress_snapshot(state, window_id="w2", enhanced=True)
    advice = evaluate_progress([first, second], enhanced=True, minimum_stop_signals=1)
    assert advice["action"] == "change_strategy"

    third = progress_snapshot(state, window_id="w3", enhanced=True)
    drain = evaluate_progress(
        [first, second, third], enhanced=True, minimum_stop_signals=1
    )
    assert drain["action"] == "consider_drain"
