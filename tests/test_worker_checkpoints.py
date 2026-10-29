import asyncio
import multiprocessing
import os
import sqlite3
import time
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END

from deepresearch.budget import SharedFetchBudgetPool, SharedSearchBudgetPool
from deepresearch.graph import ResearchGraph
from deepresearch.plan_policy import task_runtime_seed
from deepresearch.profiles import ProfileRegistry
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import ResearchTask, TaskBudget, RunBudget
from deepresearch.state import ResearchState
from deepresearch.worker_checkpoints import WorkerCheckpointStore
from deepresearch.worker_checkpoints import window_identity


def test_journal_is_durable_scoped_and_detects_corruption(tmp_path):
    path = tmp_path / "workers.sqlite"
    store = WorkerCheckpointStore(path)
    store.save("run", "window", "a", {"evidence": [{"quote": "原文"}]})
    assert WorkerCheckpointStore(path).load("run", "window")["a"]["evidence"][0]["quote"] == "原文"
    assert store.load("other-run", "window") == {}
    assert store.load("run", "changed-plan") == {}
    with pytest.raises(ValueError, match="conflicting"):
        store.save("run", "window", "a", {"evidence": []})
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE worker_results SET payload=?", (b"corrupt",))
    with pytest.raises(ValueError, match="corrupt"):
        store.load("run", "window")


def test_window_identity_changes_with_plan_and_skill_snapshot():
    state = _state()
    original = window_identity(state)
    assert original == window_identity(dict(reversed(list(state.items()))))
    state["skill_snapshot_id"] = "snapshot-v2"
    assert window_identity(state) != original
    state["skill_snapshot_id"] = "snapshot-v1"
    state["current_tasks"][0]["objective"] = "changed objective"
    assert window_identity(state) != original


def _state():
    tasks = [ResearchTask(
        task_id=name, objective=name, research_dimension=name,
        queries=[name], depends_on=[] if name == "a" else ["a"],
        budget=TaskBudget(max_searches=3, max_fetches=3, max_wall_time_seconds=60),
    ) for name in ("a", "b")]
    return {
        "run_id": "worker-recovery", "round_no": 1, "started_at": time.time(),
        "request": {"question": "research"},
        "scope": {"question": "research", "objective": "research", "mode": "standard",
                  "profile_id": "general_research", "required_dimensions": ["a", "b"],
                  "current_date": "2026-10-27", "options": {"max_workers": 2}},
        "current_tasks": [t.model_dump(mode="json") for t in tasks],
        "task_runtime": task_runtime_seed(tasks),
        "run_budget": RunBudget(max_total_searches=6, max_total_fetches=6,
                                max_wall_time_seconds=600).model_dump(mode="json"),
        "sources": [], "claims": [], "evidence": [], "usage_events": [],
        "skill_snapshot_id": "snapshot-v1",
    }


def _runner(path, worker):
    graph = object.__new__(ResearchGraph)
    graph.planner = SimpleNamespace(runtime_config=load_runtime_config())
    graph.worker_checkpoint_store = WorkerCheckpointStore(path)
    graph.worker = SimpleNamespace(search_budget_pool=SharedSearchBudgetPool(),
                                   fetch_budget_pool=SharedFetchBudgetPool())
    graph.profiles = ProfileRegistry(runtime_config=graph.planner.runtime_config)
    graph._branch_state = lambda state, task: {**state, "current_task": task}
    graph._worker_node = worker
    return graph


def _result(name):
    return {"contributions": [{"task_id": name, "status": "completed", "usable": True,
                                "source_ids": ["src-" + name]}],
            "sources": [{"source_id": "src-" + name, "canonical_uri": "https://example.com/" + name}],
            "usage_events": [{"usage_event_id": "usage-" + name, "task_id": name,
                              "searches": 1, "fetches": 1}]}


@pytest.mark.asyncio
async def test_sqlite_graph_resume_skips_completed_worker_and_restores_its_artifacts(tmp_path):
    calls = []
    b_started = asyncio.Event()
    b_cancelled = asyncio.Event()

    async def first_worker(branch):
        name = branch["current_task"]["task_id"]
        calls.append(name)
        if name == "b":
            b_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                b_cancelled.set()
        return _result(name)

    path = tmp_path / "workers.sqlite"
    db = tmp_path / "graph.sqlite"
    config = {"configurable": {"thread_id": "worker-recovery"}}
    runner = _runner(path, first_worker)
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        builder = StateGraph(ResearchState)
        builder.add_node("frontier", runner._run_frontier_window)
        builder.add_edge(START, "frontier")
        builder.add_edge("frontier", END)
        graph = builder.compile(checkpointer=saver)
        operation = asyncio.create_task(graph.ainvoke(_state(), config))
        await asyncio.wait_for(b_started.wait(), 5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await asyncio.wait_for(b_cancelled.wait(), 1)

    # Fresh graph, fresh SQLite connections and fresh in-memory budget pools.
    async def resumed_worker(branch):
        name = branch["current_task"]["task_id"]
        calls.append(name)
        assert name == "b"
        assert any(s["source_id"] == "src-a" for s in branch["sources"])
        assert branch["usage_events"][0]["usage_event_id"] == "usage-a"
        # A used one of six searches: a fresh process must expose five, not six.
        pool = runner.worker.search_budget_pool
        assert await pool.acquire("worker-recovery", 1, "b", units=5)
        assert not await pool.acquire("worker-recovery", 1, "b")
        return _result(name)

    runner = _runner(path, resumed_worker)
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        builder = StateGraph(ResearchState)
        builder.add_node("frontier", runner._run_frontier_window)
        builder.add_edge(START, "frontier")
        builder.add_edge("frontier", END)
        graph = builder.compile(checkpointer=saver)
        result = await graph.ainvoke(None, config)
        assert not (await graph.aget_state(config)).next
    assert calls == ["a", "b", "b"]
    assert {s["source_id"] for s in result["sources"]} == {"src-a", "src-b"}
    assert len(result["usage_events"]) == 2
    assert result["frontier_history"][-1]["restored_task_ids"] == ["a"]
    assert all(t["status"] == "completed" for t in result["task_runtime"])


def _crashing_process(root):
    from pathlib import Path
    root = Path(root)

    async def run():
        async def worker(branch):
            name = branch["current_task"]["task_id"]
            if name == "b":
                # No cleanup/finally: simulate the process dying mid-Worker.
                os._exit(23)
            return _result(name)

        runner = _runner(root / "workers.sqlite", worker)
        async with AsyncSqliteSaver.from_conn_string(str(root / "graph.sqlite")) as saver:
            builder = StateGraph(ResearchState)
            builder.add_node("frontier", runner._run_frontier_window)
            builder.add_edge(START, "frontier")
            builder.add_edge("frontier", END)
            await builder.compile(checkpointer=saver).ainvoke(
                _state(), {"configurable": {"thread_id": "worker-recovery"}},
            )
    asyncio.run(run())


@pytest.mark.asyncio
async def test_hard_process_exit_recovers_completed_worker(tmp_path):
    process = multiprocessing.get_context("spawn").Process(
        target=_crashing_process, args=(str(tmp_path),),
    )
    process.start()
    await asyncio.to_thread(process.join, 15)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("crash simulation did not reach the second Worker")
    assert process.exitcode == 23
    calls = []

    async def worker(branch):
        calls.append(branch["current_task"]["task_id"])
        assert {s["source_id"] for s in branch["sources"]} == {"src-a"}
        return _result("b")

    runner = _runner(tmp_path / "workers.sqlite", worker)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "graph.sqlite")) as saver:
        builder = StateGraph(ResearchState)
        builder.add_node("frontier", runner._run_frontier_window)
        builder.add_edge(START, "frontier")
        builder.add_edge("frontier", END)
        output = await builder.compile(checkpointer=saver).ainvoke(
            None, {"configurable": {"thread_id": "worker-recovery"}},
        )
    assert calls == ["b"]
    assert len(output["usage_events"]) == 2
    assert output["frontier_history"][-1]["restored_task_ids"] == ["a"]


@pytest.mark.asyncio
async def test_all_saved_workers_are_rehydrated_without_execution(tmp_path):
    state = _state()
    calls = []

    async def worker(branch):
        name = branch["current_task"]["task_id"]
        calls.append(name)
        return _result(name)

    runner = _runner(tmp_path / "workers.sqlite", worker)
    await runner._run_frontier_window(state)
    assert calls == ["a", "b"]
    calls.clear()
    runner = _runner(tmp_path / "workers.sqlite", worker)
    output = await runner._run_frontier_window(state)
    assert calls == []
    assert len(output["sources"]) == 2
    assert len(output["usage_events"]) == 2
    assert output["wave_no"] == 2
    assert output["frontier_history"][-1]["completed_count"] == 2
    assert output["frontier_history"][-1]["restored_task_ids"] == ["a", "b"]
