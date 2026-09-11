"""Concurrency-safe structured progress events for CLI and future UIs."""

from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .schemas import ProgressEvent
from .time_utils import now_iso


_CURRENT_REPORTER: contextvars.ContextVar["ProgressReporter | None"] = (
    contextvars.ContextVar("deepresearch_progress_reporter", default=None)
)


class ProgressReporter:
    def __init__(
        self,
        *,
        timezone: str = "Asia/Shanghai",
        session_id: str | None = None,
    ) -> None:
        self.timezone = timezone
        self.session_id = session_id
        self.turn_id: str | None = None
        self.run_id: str | None = None
        self.started_at = time.monotonic()
        self.queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self.history: list[ProgressEvent] = []
        self._sequence = 0
        self._active: dict[str, ProgressEvent] = {}
        self._workers: dict[int, dict[str, str]] = {}

    def bind_ids(
        self,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if session_id:
            self.session_id = session_id
        if turn_id:
            self.turn_id = turn_id
        if run_id:
            self.run_id = run_id

    def emit(
        self,
        event_type: str,
        *,
        phase: str,
        agent_role: str,
        status: str,
        message: str,
        visibility: str = "normal",
        round_no: int | None = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        operation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> ProgressEvent:
        self._sequence += 1
        event = ProgressEvent(
            event_id=f"progress-{self._sequence:06d}",
            sequence=self._sequence,
            timestamp=now_iso(self.timezone),
            elapsed_seconds=round(time.monotonic() - self.started_at, 3),
            event_type=event_type,
            phase=phase,
            agent_role=agent_role,
            status=status,
            message=message,
            visibility=visibility,
            session_id=self.session_id,
            turn_id=self.turn_id,
            run_id=run_id or self.run_id,
            round_no=round_no,
            task_id=task_id,
            worker_id=worker_id,
            operation_id=operation_id,
            metrics=metrics or {},
        )
        key = operation_id or self._activity_key(event)
        if status in {"started", "progress"}:
            self._active[key] = event
        elif status in {"completed", "failed"}:
            self._active.pop(key, None)
        self.history.append(event)
        self.queue.put_nowait(event)
        return event

    def heartbeat(self) -> ProgressEvent | None:
        if not self._active:
            return None
        active = max(self._active.values(), key=lambda item: item.sequence)
        waited = max(0.0, time.monotonic() - self.started_at - active.elapsed_seconds)
        return self.emit(
            f"{active.event_type}_heartbeat",
            phase=active.phase,
            agent_role=active.agent_role,
            status="heartbeat",
            message=f"{active.message}，仍在进行中，已等待 {waited:.0f} 秒",
            visibility="normal",
            round_no=active.round_no,
            task_id=active.task_id,
            worker_id=active.worker_id,
            operation_id=active.operation_id,
            metrics={**active.metrics, "waiting_seconds": round(waited, 1)},
        )

    def register_workers(self, round_no: int, task_ids: list[str]) -> dict[str, int]:
        self._workers[round_no] = {task_id: "pending" for task_id in task_ids}
        return self.worker_metrics(round_no)

    def mark_worker(
        self, round_no: int, task_id: str, status: str
    ) -> dict[str, int]:
        workers = self._workers.setdefault(round_no, {})
        workers[task_id] = status
        return self.worker_metrics(round_no)

    def worker_metrics(self, round_no: int) -> dict[str, int]:
        values = list(self._workers.get(round_no, {}).values())
        terminal = {"completed", "insufficient", "failed", "timed_out"}
        return {
            "total_workers": len(values),
            "running_workers": values.count("running"),
            "pending_workers": values.count("pending"),
            "finished_workers": sum(value in terminal for value in values),
            "successful_workers": values.count("completed"),
            "insufficient_workers": values.count("insufficient"),
            "failed_workers": values.count("failed") + values.count("timed_out"),
        }

    @staticmethod
    def _activity_key(event: ProgressEvent) -> str:
        return ":".join(
            str(item or "-")
            for item in (
                event.phase,
                event.agent_role,
                event.worker_id,
                event.task_id,
            )
        )


@contextmanager
def progress_context(reporter: ProgressReporter) -> Iterator[ProgressReporter]:
    token = _CURRENT_REPORTER.set(reporter)
    try:
        yield reporter
    finally:
        _CURRENT_REPORTER.reset(token)


def current_reporter() -> ProgressReporter | None:
    return _CURRENT_REPORTER.get()


def bind_progress_ids(**values: str | None) -> None:
    reporter = current_reporter()
    if reporter is not None:
        reporter.bind_ids(**values)


def emit_progress(event_type: str, **values: Any) -> ProgressEvent | None:
    reporter = current_reporter()
    if reporter is None:
        return None
    return reporter.emit(event_type, **values)
