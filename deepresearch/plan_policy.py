"""Deterministic validation and scheduling for planner-generated task DAGs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .schemas import ResearchTask


_CAPABILITY_BY_KIND = {
    "gather": "web_research",
    "analyze": "source_analysis",
    "verify": "claim_verification",
}


@dataclass(frozen=True)
class PlanValidation:
    accepted: bool
    errors: list[str]
    depth: int
    topology: list[str]
    fallback_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "errors": self.errors,
            "depth": self.depth,
            "topology": self.topology,
            "fallback_used": self.fallback_used,
        }


def _topology(tasks: list[ResearchTask]) -> tuple[list[str], int, list[str]]:
    by_id = {task.task_id: task for task in tasks}
    errors: list[str] = []
    successors: dict[str, list[str]] = defaultdict(list)
    indegree = {task.task_id: 0 for task in tasks}
    depth = {task.task_id: 1 for task in tasks}
    for task in tasks:
        for dependency in task.depends_on:
            if dependency == task.task_id:
                errors.append(f"self_dependency:{task.task_id}")
            elif dependency not in by_id:
                errors.append(f"missing_dependency:{task.task_id}:{dependency}")
            else:
                successors[dependency].append(task.task_id)
                indegree[task.task_id] += 1
    if errors:
        return [], 0, errors

    queue = deque(sorted(task_id for task_id, value in indegree.items() if value == 0))
    topology: list[str] = []
    while queue:
        task_id = queue.popleft()
        topology.append(task_id)
        for successor in sorted(successors.get(task_id, [])):
            depth[successor] = max(depth[successor], depth[task_id] + 1)
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if len(topology) != len(tasks):
        return [], 0, ["cycle_detected"]
    return topology, max(depth.values(), default=0), []


def validate_plan(
    tasks: list[ResearchTask], *, max_depth: int
) -> tuple[list[ResearchTask], PlanValidation]:
    """Validate an untrusted task DAG and fall back to safe gather tasks."""

    errors: list[str] = []
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_task_id")
    for task in tasks:
        if task.capability != _CAPABILITY_BY_KIND[task.task_kind]:
            errors.append(f"invalid_capability:{task.task_id}")
        if task.task_kind in {"analyze", "verify"} and not (
            task.depends_on or task.consumes_requirement_ids
        ):
            errors.append(f"unanchored_{task.task_kind}:{task.task_id}")
    topology, depth, topology_errors = _topology(tasks)
    errors.extend(topology_errors)
    if depth > max_depth:
        errors.append(f"depth_exceeded:{depth}>{max_depth}")
    if not errors:
        return tasks, PlanValidation(True, [], depth, topology)

    fallback = [
        task.model_copy(
            update={
                "task_id": (
                    task.task_id
                    if task.task_id not in {
                        candidate.task_id for candidate in tasks[:index]
                    }
                    else f"{task.task_id}-fallback-{index + 1}"
                ),
                "task_kind": "gather",
                "depends_on": [],
                "consumes_requirement_ids": [],
                "produces_requirement_ids": task.requirement_ids,
                "capability": "web_research",
                "produces_requirement_ids": task.requirement_ids,
            }
        )
        for index, task in enumerate(tasks)
    ]
    fallback_topology = [task.task_id for task in fallback]
    return fallback, PlanValidation(
        False,
        sorted(set(errors)),
        1 if fallback else 0,
        fallback_topology,
        fallback_used=True,
    )


def task_runtime_seed(tasks: list[ResearchTask]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task.task_id,
            "status": "pending",
            "wave_no": 0,
            "evidence_delta": 0,
            "source_delta": 0,
            "requirement_support_delta": 0,
            "produced_artifact_roles": [],
            "produced_artifact_ids": [],
            "stop_reason": "",
        }
        for task in tasks
    ]


def ready_tasks(
    tasks: list[ResearchTask], runtime: list[dict[str, Any]]
) -> tuple[list[ResearchTask], list[dict[str, Any]]]:
    """Return dependency-ready tasks and deterministic blocked updates.

    A task can finish with useful but incomplete evidence.  That state must be
    distinguishable from a task that produced nothing: downstream workers may
    consume the partial material, while requirement sufficiency remains
    tracked separately by the evidence ledger.
    """

    runtime_by_id = {str(item.get("task_id")): item for item in runtime}
    status_by_id = {
        task_id: str(item.get("status", "pending"))
        for task_id, item in runtime_by_id.items()
    }
    ready: list[ResearchTask] = []
    blocked: list[dict[str, Any]] = []
    for task in tasks:
        if status_by_id.get(task.task_id, "pending") != "pending":
            continue
        dependency_statuses = [status_by_id.get(item, "blocked") for item in task.depends_on]
        dependency_usable: list[bool] = []
        for dependency_id in task.depends_on:
            dependency_runtime = runtime_by_id.get(dependency_id, {})
            status = status_by_id.get(dependency_id)
            usable_status = bool(dependency_runtime.get("dependency_usable")) or status in {
                "completed",
                "partial",
                "completed_partial",
            }
            dependency_usable.append(usable_status)
        hard_unusable = any(
            status in {"failed", "blocked", "superseded"}
            or (status == "insufficient" and not usable)
            for status, usable in zip(dependency_statuses, dependency_usable)
        )
        all_dependencies_terminal = all(
            status in {"completed", "partial", "completed_partial", "insufficient"}
            for status in dependency_statuses
        )
        if hard_unusable:
            blocked.append(
                {
                    "task_id": task.task_id,
                    "status": "blocked",
                    "wave_no": 0,
                    "evidence_delta": 0,
                    "source_delta": 0,
                    "requirement_support_delta": 0,
                    "stop_reason": "dependency_unusable",
                }
            )
        elif all_dependencies_terminal:
            # Artifact roles guide context projection and Worker interpretation;
            # they are not an execution gate. Model-produced role labels are
            # necessarily incomplete on difficult pages, while the upstream
            # source/evidence ledger can still be useful to gather, verify, or
            # analyze tasks. Only an actually unusable dependency blocks work.
            ready.append(task)
    return ready, blocked


def select_wave(
    ready: list[ResearchTask],
    *,
    wave_size: int,
    source_origin_first: bool,
    priority_order: list[str] | None = None,
) -> list[ResearchTask]:
    """Use an LLM order when supplied; retain a stable fail-soft fallback."""

    model_order = {
        task_id: index for index, task_id in enumerate(priority_order or [])
    }

    def rank(task: ResearchTask) -> tuple[int, float, int, str]:
        if task.task_id in model_order:
            return (0, float(model_order[task.task_id]), task.priority, task.task_id)
        origin_bonus = 2.0 if source_origin_first and task.source_origin else 0.0
        kind_bonus = {"gather": 0.2, "analyze": 0.1, "verify": 0.0}[task.task_kind]
        return (
            1,
            -(task.expected_value + origin_bonus + kind_bonus),
            task.priority,
            task.task_id,
        )

    return sorted(ready, key=rank)[: max(1, wave_size)]


def estimate_remaining_waves(
    tasks: list[ResearchTask],
    runtime: list[dict[str, Any]],
    *,
    wave_size: int,
    source_origin_first: bool,
) -> int:
    """Simulate the deterministic scheduler to estimate unfinished waves.

    The simulation assumes each future task returns dependency-usable material.
    It does not relax evidence sufficiency or mutate the real runtime ledger.
    This is deliberately topology-aware: a serial source-origin task reserves
    time for every downstream level, while independent tasks share a wave.
    """

    simulated = {
        str(item.get("task_id")): dict(item)
        for item in runtime
        if item.get("task_id")
    }
    waves = 0
    while waves <= len(tasks):
        ready, _ = ready_tasks(tasks, list(simulated.values()))
        selected = select_wave(
            ready,
            wave_size=wave_size,
            source_origin_first=source_origin_first,
        )
        if not selected:
            break
        waves += 1
        for task in selected:
            current = dict(simulated.get(task.task_id, {}))
            current.update(
                {
                    "task_id": task.task_id,
                    "status": "completed",
                    "dependency_usable": True,
                }
            )
            simulated[task.task_id] = current
    return waves
