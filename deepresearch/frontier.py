"""Dependency-aware Frontier registry and continuous slot-refill scheduler."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .plan_policy import ready_tasks, select_wave
from .schemas import ResearchTask


@dataclass
class FrontierCommit:
    runtime_update: dict[str, Any]
    new_tasks: list[ResearchTask] = field(default_factory=list)
    state_patch: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontierRunResult:
    results: list[tuple[ResearchTask, dict[str, Any]]]
    runtime: list[dict[str, Any]]
    tasks: list[ResearchTask]
    state_patches: list[dict[str, Any]]
    max_running: int
    launch_order: list[str]
    completion_order: list[str]
    wave_task_ids: dict[int, list[str]]


def artifact_roles_from_result(result: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    roles: set[str] = set()
    artifact_ids: set[str] = set()
    for item in result.get("source_leads", []):
        role = str(item.get("artifact_role") or "")
        # A SourceLead describes a page/lead role; it is never by itself a
        # materialized target entity or field-evidence artifact.
        if role in {"source_document", "entity_container", "navigation_only"}:
            roles.add(role)
        if item.get("lead_id"):
            artifact_ids.add(str(item["lead_id"]))
    if result.get("sources"):
        roles.add("source_document")
        artifact_ids.update(
            str(item.get("source_id"))
            for item in result.get("sources", [])
            if item.get("source_id")
        )
    if result.get("entity_candidates"):
        materialized = [
            item
            for item in result.get("entity_candidates", [])
            if item.get("materialization_status") == "materialized"
        ]
        if materialized:
            roles.add("target_entity")
            artifact_ids.update(
                str(item.get("entity_id"))
                for item in materialized
                if item.get("entity_id")
            )
    if result.get("evidence_bindings"):
        roles.add("field_evidence")
        artifact_ids.update(
            str(item.get("binding_id"))
            for item in result.get("evidence_bindings", [])
            if item.get("binding_id")
        )
    return sorted(roles), sorted(artifact_ids)


class FrontierRegistry:
    """Mutable runtime registry; task semantics remain Planner-owned."""

    def __init__(
        self,
        tasks: Iterable[ResearchTask],
        runtime: Iterable[Mapping[str, Any]],
        *,
        enhanced: bool = False,
        max_pending_tasks: int | None = None,
    ) -> None:
        self.enhanced = enhanced
        self.max_pending_tasks = max_pending_tasks
        self.tasks: dict[str, ResearchTask] = {}
        self._signature_to_task_id: dict[str, str] = {}
        self.runtime = {
            str(item.get("task_id")): dict(item)
            for item in runtime
            if item.get("task_id")
        }
        self.add_tasks(tasks)

    @staticmethod
    def semantic_signature(task: ResearchTask) -> str:
        """Stable coarse semantic identity without replacing Planner judgment."""

        def normalized(value: str) -> str:
            return " ".join(
                re.findall(r"[\w\u3400-\u9fff]+", value.casefold(), flags=re.UNICODE)
            )

        parts = [
            task.task_kind,
            normalized(task.research_object),
            normalized(task.research_dimension),
            normalized(task.objective),
            ",".join(sorted(task.requirement_ids)),
            ",".join(sorted(task.produces_requirement_ids)),
            ",".join(sorted(task.target_entity_ids)),
            ",".join(sorted(task.target_field_keys)),
        ]
        return "|".join(parts)

    @staticmethod
    def _merge_duplicate(current: ResearchTask, incoming: ResearchTask) -> ResearchTask:
        """Merge discovery detail and bump value without changing task identity."""

        return current.model_copy(
            update={
                "queries": list(dict.fromkeys([*current.queries, *incoming.queries])),
                "key_questions": list(
                    dict.fromkeys([*current.key_questions, *incoming.key_questions])
                ),
                "preferred_source_types": list(
                    dict.fromkeys(
                        [
                            *current.preferred_source_types,
                            *incoming.preferred_source_types,
                        ]
                    )
                ),
                "depends_on": list(
                    dict.fromkeys([*current.depends_on, *incoming.depends_on])
                ),
                "requirement_ids": list(
                    dict.fromkeys([*current.requirement_ids, *incoming.requirement_ids])
                ),
                "consumes_requirement_ids": list(
                    dict.fromkeys(
                        [
                            *current.consumes_requirement_ids,
                            *incoming.consumes_requirement_ids,
                        ]
                    )
                ),
                "produces_requirement_ids": list(
                    dict.fromkeys(
                        [
                            *current.produces_requirement_ids,
                            *incoming.produces_requirement_ids,
                        ]
                    )
                ),
                "priority": min(current.priority, incoming.priority),
                "expected_value": max(current.expected_value, incoming.expected_value),
                "budget": current.budget.model_copy(
                    update={
                        "max_searches": max(
                            current.budget.max_searches, incoming.budget.max_searches
                        ),
                        "max_fetches": max(
                            current.budget.max_fetches, incoming.budget.max_fetches
                        ),
                        "max_wall_time_seconds": max(
                            current.budget.max_wall_time_seconds,
                            incoming.budget.max_wall_time_seconds,
                        ),
                    }
                ),
            }
        )

    def _cycle_task_ids(self) -> set[str]:
        visiting: set[str] = set()
        visited: set[str] = set()
        cyclic: set[str] = set()

        def visit(task_id: str, path: list[str]) -> None:
            if task_id in visiting:
                start = path.index(task_id) if task_id in path else 0
                cyclic.update(path[start:])
                return
            if task_id in visited or task_id not in self.tasks:
                return
            visiting.add(task_id)
            path.append(task_id)
            for dependency_id in self.tasks[task_id].depends_on:
                visit(dependency_id, path)
            path.pop()
            visiting.discard(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id, [])
        return cyclic

    def _enforce_queue_cap(self) -> None:
        if not self.max_pending_tasks or self.max_pending_tasks < 1:
            return
        pending = self.pending()
        overflow = len(pending) - self.max_pending_tasks
        if overflow <= 0:
            return
        depended_on = {
            dependency_id
            for task in self.tasks.values()
            for dependency_id in task.depends_on
        }
        eviction_order = sorted(
            pending,
            key=lambda task: (
                task.task_id in depended_on,
                task.expected_value,
                -task.priority,
                task.task_id,
            ),
        )
        for task in eviction_order[:overflow]:
            self.update_runtime(
                {
                    "task_id": task.task_id,
                    "status": "superseded",
                    "dependency_usable": False,
                    "stop_reason": "frontier_queue_eviction",
                }
            )

    def add_tasks(self, tasks: Iterable[ResearchTask]) -> None:
        for task in tasks:
            if self.enhanced:
                signature = self.semantic_signature(task)
                existing_id = self._signature_to_task_id.get(signature)
                if existing_id and existing_id in self.tasks:
                    current = self.tasks[existing_id]
                    self.tasks[existing_id] = self._merge_duplicate(current, task)
                    current_runtime = self.runtime.get(existing_id, {})
                    self.runtime[existing_id] = {
                        **current_runtime,
                        "merged_task_ids": list(
                            dict.fromkeys(
                                [
                                    *current_runtime.get("merged_task_ids", []),
                                    task.task_id,
                                ]
                            )
                        ),
                        "semantic_signature": signature,
                    }
                    continue
                self._signature_to_task_id[signature] = task.task_id
            self.tasks.setdefault(task.task_id, task)
            self.runtime.setdefault(
                task.task_id,
                {
                    "task_id": task.task_id,
                    "status": "pending",
                    "produced_artifact_roles": [],
                    "produced_artifact_ids": [],
                    **(
                        {"semantic_signature": self.semantic_signature(task)}
                        if self.enhanced
                        else {}
                    ),
                },
            )
        if self.enhanced:
            for task_id in self._cycle_task_ids():
                self.update_runtime(
                    {
                        "task_id": task_id,
                        "status": "blocked",
                        "dependency_usable": False,
                        "stop_reason": "frontier_cycle_detected",
                    }
                )
            self._enforce_queue_cap()

    def update_runtime(self, update: Mapping[str, Any]) -> None:
        task_id = str(update.get("task_id") or "")
        if not task_id:
            return
        self.runtime[task_id] = {**self.runtime.get(task_id, {}), **dict(update)}

    def ready(self) -> list[ResearchTask]:
        ready, blocked = ready_tasks(
            list(self.tasks.values()), list(self.runtime.values())
        )
        for item in blocked:
            self.update_runtime(item)
        return ready

    def pending(self) -> list[ResearchTask]:
        return [
            task
            for task_id, task in self.tasks.items()
            if self.runtime.get(task_id, {}).get("status", "pending") == "pending"
        ]


async def run_continuous_frontier(
    *,
    tasks: Iterable[ResearchTask],
    runtime: Iterable[Mapping[str, Any]],
    max_concurrency: int,
    execute: Callable[[ResearchTask], Awaitable[dict[str, Any]]],
    commit: Callable[
        [ResearchTask, dict[str, Any]], Awaitable[FrontierCommit] | FrontierCommit
    ],
    priority_order: list[str] | None = None,
    may_launch: Callable[[], bool] | None = None,
    enhanced: bool = False,
    max_pending_tasks: int | None = None,
) -> FrontierRunResult:
    """Merge each completion immediately and refill the freed slot."""

    registry = FrontierRegistry(
        tasks,
        runtime,
        enhanced=enhanced,
        max_pending_tasks=max_pending_tasks,
    )
    running: dict[asyncio.Task[dict[str, Any]], ResearchTask] = {}
    results: list[tuple[ResearchTask, dict[str, Any]]] = []
    state_patches: list[dict[str, Any]] = []
    launch_order: list[str] = []
    completion_order: list[str] = []
    launch_wave_by_task: dict[str, int] = {}
    wave_task_ids: dict[int, list[str]] = {}
    max_running = 0

    try:
        while running or registry.pending():
            ready = [task for task in registry.ready() if task not in running.values()]
            selected = select_wave(
                ready,
                wave_size=max(1, max_concurrency - len(running)),
                source_origin_first=False,
                priority_order=priority_order,
            )
            while selected and len(running) < max_concurrency and (
                may_launch is None or may_launch()
            ):
                task = selected.pop(0)
                dependency_waves = [
                    int(registry.runtime.get(dependency_id, {}).get("wave_no", 0))
                    for dependency_id in task.depends_on
                ]
                launch_wave = max(dependency_waves, default=0) + 1
                registry.update_runtime(
                    {
                        "task_id": task.task_id,
                        "status": "running",
                        "wave_no": launch_wave,
                    }
                )
                future = asyncio.create_task(execute(task), name=f"frontier:{task.task_id}")
                running[future] = task
                launch_order.append(task.task_id)
                launch_wave_by_task[task.task_id] = launch_wave
                wave_task_ids.setdefault(launch_wave, []).append(task.task_id)
                max_running = max(max_running, len(running))

            if not running:
                # Remaining tasks are blocked or launch policy entered drain mode.
                break

            done, _ = await asyncio.wait(
                running.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                task = running.pop(future)
                try:
                    result = future.result()
                except asyncio.CancelledError:
                    result = {"error": "cancelled", "contributions": []}
                except Exception as exc:  # The graph records the concrete error.
                    result = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "contributions": [],
                    }
                committed = commit(task, result)
                if isinstance(committed, Awaitable):
                    committed = await committed
                registry.update_runtime(
                    {
                        **committed.runtime_update,
                        "wave_no": launch_wave_by_task[task.task_id],
                    }
                )
                registry.add_tasks(committed.new_tasks)
                if committed.state_patch:
                    state_patches.append(committed.state_patch)
                results.append((task, result))
                completion_order.append(task.task_id)

    finally:
        # Parent cancellation or a failed durable commit must not leave orphan
        # Workers executing tools after their scheduler has stopped.
        for future in running:
            future.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    return FrontierRunResult(
        results=results,
        runtime=list(registry.runtime.values()),
        tasks=list(registry.tasks.values()),
        state_patches=state_patches,
        max_running=max_running,
        launch_order=launch_order,
        completion_order=completion_order,
        wave_task_ids=wave_task_ids,
    )
