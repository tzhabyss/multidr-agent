from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deepresearch.interactive import (
    ProgressRenderer,
    terminal_safe_text,
    terminal_supports_unicode_markdown,
)
from deepresearch.progress import ProgressReporter, emit_progress
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import (
    ProgressEvent,
    ResearchConstraints,
    ResearchOptions,
    ResearchRequest,
    RunResult,
    SessionMemory,
)
from deepresearch.service import ResearchService
from deepresearch.time_utils import convert_iso
from tests.fakes import FakeFetcher, FakeLLM, FakeSearchBackend
from tests.test_interactive import CapturedOutput
from tests.test_memory_session import SessionAwareFakeLLM


def _runtime(tmp_path: Path):
    config = load_runtime_config()
    return config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={
                    "local_db_path": str(tmp_path / "memory.sqlite"),
                    "automatic_capture_enabled": False,
                }
            )
        }
    )


def test_project_timestamps_default_to_china_standard_time():
    session = SessionMemory(session_id="session-timezone-test")
    event = ProgressReporter().emit(
        "timezone_test",
        phase="test",
        agent_role="test",
        status="completed",
        message="done",
    )

    assert session.created_at.endswith("+08:00")
    assert session.updated_at.endswith("+08:00")
    assert event.timestamp.endswith("+08:00")
    assert convert_iso("2026-07-21T08:00:00+00:00") == (
        "2026-07-21T16:00:00+08:00"
    )


@pytest.mark.asyncio
async def test_chat_stream_exposes_every_user_visible_phase(tmp_path: Path):
    events: list[ProgressEvent] = []
    final = None
    async with ResearchService.in_memory(
        llm=SessionAwareFakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=_runtime(tmp_path),
    ) as service:
        async for item in service.chat_stream(
            "What is the verified fact?", heartbeat_seconds=0
        ):
            if isinstance(item, ProgressEvent):
                events.append(item)
            else:
                final = item

    event_types = {item.event_type for item in events}
    assert final is not None and final.run is not None
    assert {
        "memory_retrieval_started",
        "clarifier_started",
        "clarifier_completed",
        "planner_started",
        "planner_completed",
        "worker_started",
        "worker_strategy_started",
        "worker_strategy_completed",
        "agent_loop_started",
        "agent_loop_completed",
        "search_completed",
        "fetch_started",
        "fetch_completed",
        "extraction_started",
        "extraction_completed",
        "worker_reflection_started",
        "worker_reflection_completed",
        "worker_completed",
        "synthesizer_started",
        "synthesizer_completed",
        "writer_started",
        "writer_completed",
        "citations_completed",
        "artifacts_started",
        "artifacts_completed",
        "memory_sync_started",
        "memory_sync_completed",
        "run_completed",
        "session_completed",
    } <= event_types
    assert all(item.timestamp.endswith("+08:00") for item in events)
    assert all(source.fetched_at.endswith("+08:00") for source in final.run.sources)
    durable_events = [
        json.loads(line)
        for line in (
            Path(final.run.artifact_dir) / "details" / "runtime" / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert durable_events
    assert all(item["timestamp"].endswith("+08:00") for item in durable_events)
    assert [item.sequence for item in events] == sorted(
        item.sequence for item in events
    )
    loop_done = next(item for item in events if item.event_type == "agent_loop_completed")
    assert loop_done.metrics["stop_reason"] == "tool_calling_unavailable"


@pytest.mark.asyncio
async def test_parallel_worker_progress_is_counted_by_task_not_event_order(tmp_path: Path):
    events: list[ProgressEvent] = []
    final: RunResult | None = None
    request = ResearchRequest(
        question="Compare three independently researched dimensions",
        constraints=ResearchConstraints(
            required_dimensions=["dimension-a", "dimension-b", "dimension-c"]
        ),
        options=ResearchOptions(max_rounds=1, max_workers=3),
    )
    async with ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=_runtime(tmp_path),
    ) as service:
        async for item in service.run_stream(request, heartbeat_seconds=0):
            if isinstance(item, ProgressEvent):
                events.append(item)
            else:
                final = item

    plan = next(item for item in events if item.event_type == "planner_completed")
    completed = [item for item in events if item.event_type == "worker_completed"]
    assert plan.metrics["total_workers"] == 3
    assert [item["worker_id"] for item in plan.metrics["tasks"]] == [
        "R1-W1",
        "R1-W2",
        "R1-W3",
    ]
    assert len(completed) == 3
    assert sorted(item.metrics["finished_workers"] for item in completed) == [1, 2, 3]
    assert completed[-1].metrics["running_workers"] == 0
    assert final is not None


@pytest.mark.asyncio
async def test_need_more_stream_opens_a_new_planner_and_worker_round(tmp_path: Path):
    events: list[ProgressEvent] = []
    async with ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(fail_first=True),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=_runtime(tmp_path),
    ) as service:
        async for item in service.run_stream(
            ResearchRequest(question="Retry this streamed research"),
            heartbeat_seconds=0,
        ):
            if isinstance(item, ProgressEvent):
                events.append(item)

    plans = [item for item in events if item.event_type == "planner_completed"]
    decisions = [
        item.metrics["decision"]
        for item in events
        if item.event_type == "synthesizer_completed"
    ]
    worker_ids = {
        item.worker_id for item in events if item.event_type == "worker_started"
    }
    assert [item.round_no for item in plans] == [1, 2]
    assert decisions == ["need_more", "ready"]
    assert {"R1-W1", "R2-W1"} <= worker_ids


@pytest.mark.asyncio
async def test_stream_emits_truthful_heartbeat_for_long_operation(tmp_path: Path):
    service = ResearchService(runtime_config=_runtime(tmp_path))
    reporter = ProgressReporter()

    async def slow_operation():
        emit_progress(
            "slow_started",
            phase="test",
            agent_role="worker",
            status="started",
            message="等待测试操作",
            operation_id="slow",
        )
        await asyncio.sleep(0.07)
        emit_progress(
            "slow_completed",
            phase="test",
            agent_role="worker",
            status="completed",
            message="测试操作完成",
            operation_id="slow",
        )
        return "done"

    items = [
        item
        async for item in service._stream_with_progress(
            slow_operation, reporter=reporter, heartbeat_seconds=0.02
        )
    ]

    heartbeats = [item for item in items if isinstance(item, ProgressEvent) and item.status == "heartbeat"]
    assert heartbeats
    assert heartbeats[0].metrics["waiting_seconds"] >= 0
    assert items[-1] == "done"


def test_progress_renderer_filters_detail_and_never_emits_ansi():
    output = CapturedOutput()
    renderer = ProgressRenderer(output, "normal")
    verbose = ProgressEvent(
        event_id="progress-1",
        sequence=1,
        timestamp="2026-07-21T16:00:00+08:00",
        elapsed_seconds=5,
        event_type="search_completed",
        phase="search",
        agent_role="worker",
        status="completed",
        message="search done",
        visibility="verbose",
        metrics={
            "query": "example query",
            "providers": ["tavily"],
            "input_tokens": 10,
            "output_tokens": 5,
        },
    )
    normal = verbose.model_copy(
        update={
            "event_id": "progress-2",
            "sequence": 2,
            "visibility": "normal",
            "worker_id": "R1-W2",
        }
    )

    renderer.render(verbose)
    assert output.text == ""
    renderer.render(normal)
    assert "[16:00:00 | +00:05] R1-W2" in output.text
    assert "\x1b" not in output.text
    renderer.level = "verbose"
    renderer.render(verbose)
    assert "providers=['tavily']" in output.text
    assert "input_tokens=10" in output.text


def test_gbk_terminal_falls_back_without_unicode_rendering_failure():
    assert terminal_safe_text("• research ✓", "gbk") == "? research ?"
    assert not terminal_supports_unicode_markdown("gbk")
    assert terminal_supports_unicode_markdown("utf-8")
