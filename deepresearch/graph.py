"""LangGraph orchestration for unified, multi-round deep research."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import Planner, ResearchWorker, Synthesizer
from .artifacts import ArtifactWorkspace
from .budget import remaining_task_budget, summarize_usage, usage_event
from .evidence import apply_citation_gate, render_markdown, validate_citations
from .context_projection import project_context, projection_trace
from .frontier import (
    FrontierCommit,
    artifact_roles_from_result,
    run_continuous_frontier,
)
from .ids import make_id
from .memory import LayeredMemory
from .plan_policy import (
    estimate_remaining_waves,
    ready_tasks,
    select_wave,
    task_runtime_seed,
    validate_plan,
)
from .profiles import ProfileRegistry
from .progress import current_reporter, emit_progress
from .observation_middleware import delta_from_worker_result
from .retrieval import independent_domain
from .schemas import (
    ClaimRecord,
    ClaimEvidenceLink,
    CitationRecord,
    EvidenceRecord,
    ResearchContribution,
    MemoryPacket,
    ResearchRequest,
    ResearchScope,
    ResearchTask,
    RunBudget,
    RunEvent,
    RunStatus,
    SourceRecord,
    SourceSnapshot,
    SearchSchemaDefinition,
    StructuredOutput,
    utc_now,
)
from .search_sensors import evaluate_progress, progress_snapshot
from .skill_evolution import EvolutionEpisodeStore, SkillEvolutionObserver
from .runtime_config import load_runtime_config
from .search_state import (
    ObservationValidationError,
    calculate_dual_coverage,
    commit_observation_delta,
    prune_invalid_observation_items,
    reconcile_schema_revision,
    schema_from_plan,
)
from .state import ResearchState
from . import state as state_ops
from .time_utils import now


def _has_unattempted_search_candidates(state: ResearchState) -> bool:
    attempted_urls = {
        str(item.get("url"))
        for item in state.get("fetch_traces", [])
        if item.get("url")
        and item.get("status") in {None, "fetched", "no_evidence"}
    }
    return any(
        item.get("url") and str(item["url"]) not in attempted_urls
        for trace in state.get("search_traces", [])
        for item in trace.get("results", [])
        if isinstance(item, dict)
    )


def _research_soft_wall_time_seconds(
    state: ResearchState,
    profiles: ProfileRegistry,
) -> float:
    request = ResearchRequest.model_validate(state["request"])
    profile = profiles.get(request.profile_id)
    override = profile.runtime_policy.mode_overrides.get(request.mode.value)
    configured = (
        override.research_soft_wall_time_seconds if override is not None else None
    )
    if configured is not None:
        return float(configured)
    return float(RunBudget.model_validate(state["run_budget"]).max_wall_time_seconds)


def _required_next_round_seconds(
    round_elapsed_seconds: float,
    soft_wall_time_seconds: float,
    max_rounds: int,
) -> float:
    """Reserve a bounded allowance for a focused gap-closing round.

    A follow-up round should have enough time to issue a small search batch and
    fetch its best candidates, but it must not reserve the entire breadth-first
    round that just completed. The half-round share is deliberately bounded so
    this remains useful when a run has only a few minutes left.
    """

    elapsed = max(0.001, float(round_elapsed_seconds))
    per_round_share = float(soft_wall_time_seconds) / max(1, int(max_rounds))
    return min(elapsed, max(120.0, per_round_share * 0.5))


def _planner_execution_capacity(
    remaining_research_seconds: float,
    graph_config: Any,
    max_workers: int | None = None,
) -> tuple[int, int]:
    """Bound plan breadth by the number of meaningful Worker waves left."""

    finalization_reserve = min(
        float(graph_config.research_finalization_reserve_seconds),
        max(0.0, remaining_research_seconds) * 0.10,
    )
    executable_seconds = max(
        0.0, remaining_research_seconds - finalization_reserve
    )
    executable_waves = max(
        1,
        int(executable_seconds // float(graph_config.min_worker_launch_seconds)),
    )
    effective_wave_size = min(
        int(graph_config.wave_size),
        max(1, int(max_workers or graph_config.wave_size)),
    )
    task_capacity = min(
        graph_config.max_plan_tasks,
        max(1, effective_wave_size * executable_waves),
    )
    return task_capacity, executable_waves


class ResearchGraph:
    def __init__(
        self,
        *,
        planner: Planner,
        worker: ResearchWorker,
        synthesizer: Synthesizer,
        profiles: ProfileRegistry,
        workspace: ArtifactWorkspace,
        evolution_store: EvolutionEpisodeStore | None = None,
        checkpointer: Any,
    ):
        self.planner = planner
        self.worker = worker
        self.synthesizer = synthesizer
        self.profiles = profiles
        self.workspace = workspace
        self.evolution_store = evolution_store
        self.graph = self._build().compile(checkpointer=checkpointer)

    @staticmethod
    def _event(
        state: ResearchState,
        event_type: str,
        sequence: int,
        *,
        task_id: str | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict:
        return RunEvent(
            event_id=make_id("evt", state["run_id"], event_type, task_id, sequence),
            run_id=state["run_id"],
            sequence=sequence,
            event_type=event_type,
            task_id=task_id,
            status=status,
            payload=payload or {},
        ).model_dump()

    def _build(self) -> StateGraph:
        builder = StateGraph(ResearchState)
        from .tool_runtime import current_runtime, operation_context
        original_add_node = builder.add_node

        def add_observed_node(name, function):
            async def observed(state):
                runtime = current_runtime()
                if runtime is None:
                    return await function(state)
                task = state.get("current_task") or {}
                with operation_context(stage=name, task_id=task.get("task_id"), round_no=state.get("round_no")):
                    kind = "agent" if name in {"plan_research", "research_worker", "synthesize_research"} else "workflow"
                    return await runtime.execute(name, lambda: function(state), kind=kind)
            return original_add_node(name, observed)

        add_observed_node("prepare_request", self._prepare_request)
        add_observed_node("normalize_scope", self._normalize_scope)
        add_observed_node("plan_research", self._plan_research)
        add_observed_node("validate_plan", self._validate_plan)
        add_observed_node("schedule_wave", self._schedule_wave)
        add_observed_node("run_frontier_window", self._run_frontier_window)
        add_observed_node("research_worker", self._worker_node)
        add_observed_node("merge_wave", self._merge_wave)
        add_observed_node("deterministic_checks", self._deterministic_checks)
        add_observed_node("control_wave", self._control_wave)
        add_observed_node("synthesize_research", self._synthesize_research)
        add_observed_node("build_structured_output", self._build_output)
        add_observed_node("citation_gate", self._citation_gate)
        add_observed_node("finalize_run", self._finalize_run)
        add_observed_node("collect_skill_evolution", self._collect_skill_evolution)
        add_observed_node("persist_artifacts", self._persist_artifacts)

        builder.add_edge(START, "prepare_request")
        builder.add_edge("prepare_request", "normalize_scope")
        builder.add_edge("normalize_scope", "plan_research")
        graph_config = self.planner.runtime_config.graph_engineering
        if graph_config.enabled and graph_config.continuous_frontier_enabled:
            builder.add_edge("plan_research", "validate_plan")
            builder.add_edge("validate_plan", "run_frontier_window")
            builder.add_edge("run_frontier_window", "deterministic_checks")
            builder.add_edge("deterministic_checks", "synthesize_research")
        elif graph_config.enabled and graph_config.wave_scheduler_enabled:
            builder.add_edge("plan_research", "validate_plan")
            builder.add_edge("validate_plan", "schedule_wave")
            builder.add_conditional_edges("schedule_wave", self._dispatch_wave)
            builder.add_edge("research_worker", "merge_wave")
            builder.add_edge("merge_wave", "deterministic_checks")
            builder.add_edge("deterministic_checks", "control_wave")
            builder.add_conditional_edges("control_wave", self._route_controller)
        else:
            builder.add_conditional_edges("plan_research", self._dispatch_tasks)
            builder.add_edge("research_worker", "synthesize_research")
        builder.add_conditional_edges("synthesize_research", self._route_synthesis)
        builder.add_edge("build_structured_output", "citation_gate")
        builder.add_edge("citation_gate", "finalize_run")
        builder.add_edge("finalize_run", "collect_skill_evolution")
        builder.add_edge("collect_skill_evolution", "persist_artifacts")
        builder.add_edge("persist_artifacts", END)
        return builder

    async def _prepare_request(self, state: ResearchState) -> dict:
        request = ResearchRequest.model_validate(state["request"])
        budget = RunBudget.model_validate(state["run_budget"])
        emit_progress(
            "run_started",
            phase="run",
            agent_role="orchestrator",
            status="started",
            message=f"研究任务已启动：{state['run_id']}",
            operation_id=f"run:{state['run_id']}",
            run_id=state["run_id"],
            visibility="quiet",
        )
        return {
            "request": request.model_dump(mode="json"),
            "run_budget": budget.model_dump(),
            "status": RunStatus.RUNNING.value,
            "round_no": 0,
            "events": [
                self._event(
                    state,
                    "run_started",
                    10,
                    status=RunStatus.RUNNING.value,
                    payload={"mode": request.mode.value, "profile_id": request.profile_id},
                )
            ],
        }

    async def _normalize_scope(self, state: ResearchState) -> dict:
        request = ResearchRequest.model_validate(state["request"])
        profile = self.profiles.get(request.profile_id)
        excluded = set(request.constraints.excluded_dimensions)
        required = list(
            dict.fromkeys(
                item
                for item in request.constraints.required_dimensions
                if item not in excluded
            )
        )
        suggested = [item for item in profile.default_dimensions if item not in excluded]
        assumptions: list[str] = []
        if not request.constraints.date_boundary:
            assumptions.append("使用运行时可访问的最新公开信息，并记录抓取时间")
        scope = ResearchScope(
            question=request.question,
            mode=request.mode,
            profile_id=request.profile_id,
            objective=f"在公开来源范围内回答：{request.question}",
            required_dimensions=required,
            suggested_dimensions=suggested,
            current_date=now().date().isoformat(),
            date_boundary=request.constraints.date_boundary,
            locale=request.constraints.locale,
            assumptions=assumptions,
            source_urls=request.source_urls,
            options=request.options,
        )
        emit_progress(
            "scope_normalized",
            phase="planning",
            agent_role="planner",
            status="completed",
            message="研究范围和约束已标准化",
            visibility="verbose",
            metrics={
                "required_dimensions": len(required),
                "suggested_dimensions": len(suggested),
            },
        )
        return {
            "scope": scope.model_dump(mode="json"),
            "events": [
                self._event(
                    state,
                    "scope_normalized",
                    20,
                    payload={
                        "required_dimensions": required,
                        "suggested_dimensions": suggested,
                        "current_date": scope.current_date,
                        "date_boundary": scope.date_boundary,
                    },
                )
            ],
        }

    async def _plan_research(self, state: ResearchState) -> dict:
        round_started_at = time.time()
        scope = ResearchScope.model_validate(state["scope"])
        run_budget = RunBudget.model_validate(state["run_budget"])
        round_no = int(state.get("round_no", 0)) + 1
        operation_id = f"planner:r{round_no}"
        emit_progress(
            "planner_started",
            phase="planning",
            agent_role="planner",
            status="started",
            message=f"Planner 正在制定第 {round_no} 轮研究计划",
            operation_id=operation_id,
            round_no=round_no,
            visibility="quiet",
        )
        round_budget = remaining_task_budget(
            run_budget,
            state.get("usage_events", []),
            round_no=round_no,
            max_rounds=scope.options.max_rounds,
        )
        attempted_queries = [
            item.get("query", "")
            for item in state.get("search_traces", [])
            if item.get("query")
        ]
        evidence_summary = {
            "source_count": len(state.get("sources", [])),
            "claim_count": len(state.get("claims", [])),
            "candidate_leads": state.get("candidate_leads", []),
            "prior_plans": state.get("plan_history", []),
            "existing_claims": [
                {"claim_id": item.get("claim_id"), "text": item.get("text")}
                for item in state.get("claims", [])[-20:]
            ],
        }
        graph_config = self.planner.runtime_config.graph_engineering
        now_timestamp = time.time()
        elapsed_run_seconds = max(
            0.0, now_timestamp - float(state.get("started_at", now_timestamp))
        )
        remaining_research_seconds = max(
            0.0,
            min(
                float(run_budget.max_wall_time_seconds),
                _research_soft_wall_time_seconds(state, self.profiles),
            )
            - elapsed_run_seconds,
        )
        planner_task_capacity, executable_waves = _planner_execution_capacity(
            remaining_research_seconds,
            graph_config,
            scope.options.max_workers,
        )
        plan, tasks, usage, planner_trace = await self.planner.plan(
            state["run_id"],
            scope,
            round_budget,
            round_no=round_no,
            gaps=state.get("unresolved_gaps", []),
            attempted_queries=attempted_queries,
            evidence_summary=evidence_summary,
            memory_packet=state.get("planner_memory", {}),
            max_tasks_override=planner_task_capacity,
        )
        plan = {
            **plan,
            "starting_claim_count": len(state.get("claims", [])),
            "starting_source_count": len(state.get("sources", [])),
            "starting_covered_requirement_ids": sorted(
                {
                    str(requirement_id)
                    for claim in state.get("claims", [])
                    for requirement_id in claim.get("requirement_ids", [])
                    if str(requirement_id)
                }
            ),
            "starting_independent_domains": sorted(
                {
                    independent_domain(str(item.get("canonical_uri") or ""))
                    for item in state.get("sources", [])
                    if item.get("canonical_uri")
                }
                - {""}
            ),
            "starting_high_priority_gap_count": sum(
                1
                for item in state.get("unresolved_gaps", [])
                if str(item.get("priority", "medium")) == "high"
            ),
            "round_started_at": round_started_at,
            "planner_task_capacity": planner_task_capacity,
            "planner_remaining_research_seconds": round(
                remaining_research_seconds, 3
            ),
            "planner_executable_waves": executable_waves,
        }
        proposed_schema = schema_from_plan(
            run_id=state["run_id"],
            plan=plan,
            scope=scope.model_dump(mode="json"),
        )
        previous_schema = (
            SearchSchemaDefinition.model_validate(state["search_schemas"][-1])
            if state.get("search_schemas")
            else None
        )
        preserved_artifact_ids = [
            str(item.get(key))
            for state_key, key in (
                ("sources", "source_id"),
                ("entity_candidates", "entity_id"),
                ("evidence", "evidence_id"),
                ("evidence_bindings", "binding_id"),
            )
            for item in state.get(state_key, [])
            if item.get(key)
        ]
        search_schema, schema_revision = reconcile_schema_revision(
            previous_schema,
            proposed_schema,
            preserved_artifact_ids=preserved_artifact_ids,
        )
        plan["search_schema"] = search_schema.model_dump(mode="json")
        await self.worker.fetch_budget_pool.configure_round(
            state["run_id"],
            round_no,
            {item.task_id: item.budget.max_fetches for item in tasks},
        )
        await self.worker.search_budget_pool.configure_round(
            state["run_id"],
            round_no,
            {item.task_id: item.budget.max_searches for item in tasks},
        )
        reporter = current_reporter()
        worker_metrics = (
            reporter.register_workers(round_no, [item.task_id for item in tasks])
            if reporter is not None
            else {"total_workers": len(tasks)}
        )
        emit_progress(
            "planner_completed",
            phase="planning",
            agent_role="planner",
            status="completed",
            message=(
                f"第 {round_no} 轮计划完成：分配 {len(tasks)} 个 Worker，"
                + ("并行执行" if len(tasks) > 1 else "单任务执行")
            ),
            operation_id=operation_id,
            round_no=round_no,
            visibility="quiet",
            metrics={
                **worker_metrics,
                "max_workers": scope.options.max_workers,
                "query_type": plan["query_type"],
                "input_tokens": planner_trace.input_tokens,
                "output_tokens": planner_trace.output_tokens,
                "duration_seconds": planner_trace.duration_seconds,
                "tasks": [
                    {
                        "worker_id": f"R{round_no}-W{index}",
                        "task_id": item.task_id,
                        "dimension": item.research_dimension,
                        "objective": item.objective,
                    }
                    for index, item in enumerate(tasks, 1)
                ],
            },
        )
        return {
            "tasks": [item.model_dump(mode="json") for item in tasks],
            "current_tasks": [item.model_dump(mode="json") for item in tasks],
            "current_plan": plan,
            "plan_history": [plan],
            "round_no": round_no,
            "usage_events": usage,
            "planner_trace": planner_trace.model_dump(mode="json"),
            "planner_traces": [planner_trace.model_dump(mode="json")],
            "search_schemas": [search_schema.model_dump(mode="json")],
            "schema_revisions": (
                [schema_revision.model_dump(mode="json")]
                if schema_revision is not None
                else []
            ),
            "events": [
                self._event(
                    state,
                    "plan_created",
                    30 + (round_no - 1) * 200,
                    payload={
                        "round_no": round_no,
                        "query_type": plan["query_type"],
                        "task_count": len(tasks),
                        "topology": "single" if len(tasks) == 1 else "parallel",
                        "targeted_gaps": len(state.get("unresolved_gaps", [])),
                        "task_capacity": planner_task_capacity,
                        "remaining_research_seconds": round(
                            remaining_research_seconds, 3
                        ),
                        "executable_waves": executable_waves,
                    },
                )
            ],
        }

    async def _validate_plan(self, state: ResearchState) -> dict:
        graph_config = self.planner.runtime_config.graph_engineering
        tasks = [
            ResearchTask.model_validate(item)
            for item in state.get("current_tasks", [])
        ]
        validated, validation = validate_plan(
            tasks,
            max_depth=graph_config.max_dag_depth,
        )
        payload = validation.as_dict()
        payload["round_no"] = int(state.get("round_no", 1))
        return {
            "tasks": [item.model_dump(mode="json") for item in validated],
            "current_tasks": [item.model_dump(mode="json") for item in validated],
            "current_wave_tasks": [],
            "wave_no": 0,
            "dag_validation": payload,
            "task_runtime": task_runtime_seed(validated),
            "events": [
                self._event(
                    state,
                    "dag_validated",
                    35 + (int(state.get("round_no", 1)) - 1) * 200,
                    status="accepted" if validation.accepted else "fallback",
                    payload=payload,
                )
            ],
        }

    @staticmethod
    def _merge_runtime_updates(
        runtime: list[dict[str, Any]], updates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id = {str(item.get("task_id")): dict(item) for item in runtime}
        for item in updates:
            by_id[str(item.get("task_id"))] = dict(item)
        return list(by_id.values())

    async def _schedule_wave(self, state: ResearchState) -> dict:
        graph_config = self.planner.runtime_config.graph_engineering
        scope = ResearchScope.model_validate(state["scope"])
        tasks = [ResearchTask.model_validate(item) for item in state.get("current_tasks", [])]
        runtime = list(state.get("task_runtime", []))

        sufficient_requirements = {
            str(item.get("requirement_id"))
            for item in state.get("requirement_status", [])
            if item.get("operational_status", item.get("status")) == "sufficient"
        }
        superseded_updates = [
            {
                "task_id": task.task_id,
                "status": "superseded",
                "wave_no": int(state.get("wave_no", 0)),
                "evidence_delta": 0,
                "source_delta": 0,
                "requirement_support_delta": 0,
                "stop_reason": "requirements_already_sufficient",
            }
            for task in tasks
            if task.produces_requirement_ids
            and set(task.produces_requirement_ids) <= sufficient_requirements
            and any(
                item.get("task_id") == task.task_id
                and item.get("status") == "pending"
                for item in runtime
            )
        ]
        runtime = self._merge_runtime_updates(runtime, superseded_updates)

        # Resolve blocked chains to a fixed point before selecting runnable
        # work. This prevents a downstream task from waiting forever when an
        # upstream branch failed in an earlier wave.
        all_blocked: list[dict[str, Any]] = []
        while True:
            _, blocked = ready_tasks(tasks, runtime)
            new_blocked = [
                item
                for item in blocked
                if not any(
                    current.get("task_id") == item.get("task_id")
                    and current.get("status") == "blocked"
                    for current in runtime
                )
            ]
            if not new_blocked:
                break
            all_blocked.extend(new_blocked)
            runtime = self._merge_runtime_updates(runtime, new_blocked)

        ready, _ = ready_tasks(tasks, runtime)
        wave_size = min(graph_config.wave_size, scope.options.max_workers)
        estimated_remaining_waves = estimate_remaining_waves(
            tasks,
            runtime,
            wave_size=wave_size,
            source_origin_first=graph_config.source_origin_first,
        )
        selected = select_wave(
            ready,
            wave_size=wave_size,
            source_origin_first=graph_config.source_origin_first,
        )
        scheduled_at = time.time()
        run_budget = RunBudget.model_validate(state["run_budget"])
        elapsed_run_seconds = max(
            0.0, scheduled_at - float(state.get("started_at", scheduled_at))
        )
        soft_remaining_seconds = max(
            0.0,
            _research_soft_wall_time_seconds(state, self.profiles)
            - elapsed_run_seconds,
        )
        hard_remaining_seconds = max(
            0.0, float(run_budget.max_wall_time_seconds) - elapsed_run_seconds
        )
        remaining_research_seconds = min(
            soft_remaining_seconds, hard_remaining_seconds
        )
        research_finalization_reserve_seconds = min(
            float(graph_config.research_finalization_reserve_seconds),
            remaining_research_seconds * 0.10,
        )
        allocatable_wave_seconds = max(
            0.0,
            remaining_research_seconds - research_finalization_reserve_seconds,
        )
        per_wave_seconds = (
            allocatable_wave_seconds / estimated_remaining_waves
            if estimated_remaining_waves > 0
            else 0.0
        )
        wave_budget_seconds = min(
            float(graph_config.max_worker_wall_time_seconds), per_wave_seconds
        )
        worker_finalization_reserve_seconds = min(
            float(graph_config.worker_finalization_reserve_seconds),
            wave_budget_seconds * 0.20,
        )
        wave_deadline_at = scheduled_at + wave_budget_seconds
        wave_tool_deadline_at = max(
            scheduled_at,
            wave_deadline_at - worker_finalization_reserve_seconds,
        )
        wave_budget = {
            "scheduled_at": scheduled_at,
            "remaining_research_seconds": round(remaining_research_seconds, 3),
            "soft_remaining_seconds": round(soft_remaining_seconds, 3),
            "hard_remaining_seconds": round(hard_remaining_seconds, 3),
            "estimated_remaining_waves": estimated_remaining_waves,
            "research_finalization_reserve_seconds": round(
                research_finalization_reserve_seconds, 3
            ),
            "worker_finalization_reserve_seconds": round(
                worker_finalization_reserve_seconds, 3
            ),
            "worker_ceiling_seconds": float(
                graph_config.max_worker_wall_time_seconds
            ),
            "wave_budget_seconds": round(wave_budget_seconds, 3),
            "wave_deadline_at": wave_deadline_at,
            "wave_tool_deadline_at": wave_tool_deadline_at,
        }
        wave_no = int(state.get("wave_no", 0)) + (1 if selected else 0)
        running_updates = [
            {
                "task_id": task.task_id,
                "status": "running",
                "wave_no": wave_no,
                "evidence_delta": 0,
                "source_delta": 0,
                "requirement_support_delta": 0,
                "stop_reason": "",
                "worker_effective_timeout_seconds": round(wave_budget_seconds, 3),
                "wave_deadline_at": wave_deadline_at,
                "worker_tool_deadline_at": wave_tool_deadline_at,
            }
            for task in selected
        ]
        covered_requirements = {
            str(requirement_id)
            for claim in state.get("claims", [])
            for requirement_id in claim.get("requirement_ids", [])
            if str(requirement_id)
        }
        return {
            "current_wave_tasks": [item.model_dump(mode="json") for item in selected],
            "wave_no": wave_no,
            "wave_budget": wave_budget,
            "wave_deadline_at": wave_deadline_at,
            "wave_tool_deadline_at": wave_tool_deadline_at,
            "wave_baseline": {
                "source_count": len(state.get("sources", [])),
                "evidence_count": len(state.get("evidence", [])),
                "claim_count": len(state.get("claims", [])),
                "covered_requirement_ids": sorted(covered_requirements),
                "started_at": scheduled_at,
                "budget": wave_budget,
            },
            "task_runtime": [*superseded_updates, *all_blocked, *running_updates],
            "events": [
                self._event(
                    state,
                    "wave_scheduled",
                    40 + (int(state.get("round_no", 1)) - 1) * 200 + wave_no,
                    status="scheduled" if selected else "empty",
                    payload={
                        "round_no": int(state.get("round_no", 1)),
                        "wave_no": wave_no,
                        "task_ids": [item.task_id for item in selected],
                        "blocked_task_ids": [item["task_id"] for item in all_blocked],
                        **wave_budget,
                    },
                )
            ],
        }

    async def _run_frontier_window(self, state: ResearchState) -> dict:
        """Run ready DAG tasks continuously, committing each completion at once."""

        tasks = [
            ResearchTask.model_validate(item)
            for item in state.get("current_tasks", [])
        ]
        scope = ResearchScope.model_validate(state["scope"])
        graph_config = self.planner.runtime_config.graph_engineering
        append_only = graph_config.p0_feature_enabled("append_only_evidence")
        local_state: dict[str, Any] = dict(state)
        reducer_by_key = {
            "contributions": state_ops.merge_contributions,
            "sources": state_ops.merge_sources,
            "source_snapshots": state_ops.merge_snapshots,
            "evidence": (
                state_ops.merge_evidence_append_only
                if append_only
                else state_ops.merge_evidence
            ),
            "claims": (
                state_ops.merge_claims_append_only
                if append_only
                else state_ops.merge_claims
            ),
            "links": (
                state_ops.merge_links_append_only
                if append_only
                else state_ops.merge_links
            ),
            "usage_events": state_ops.merge_usage,
            "events": state_ops.merge_events,
            "extraction_traces": state_ops.merge_extraction_traces,
            "search_traces": state_ops.merge_search_traces,
            "fetch_traces": state_ops.merge_fetch_traces,
            "agent_loop_traces": state_ops.merge_agent_loop_traces,
            "worker_traces": state_ops.merge_worker_traces,
            "candidate_leads": state_ops.merge_candidate_leads,
            "source_leads": state_ops.merge_source_leads,
            "entity_candidates": state_ops.merge_entity_candidates,
            "evidence_bindings": (
                state_ops.merge_evidence_bindings_append_only
                if append_only
                else state_ops.merge_evidence_bindings
            ),
            "evidence_edges": (
                state_ops.merge_evidence_edges_append_only
                if append_only
                else state_ops.merge_evidence_edges
            ),
            "failure_memory": state_ops.merge_failure_memory,
            "coverage_changes": state_ops.merge_coverage_changes,
            "observation_commits": state_ops.merge_observation_commits,
            "observation_rejections": state_ops.merge_observation_rejections,
            "context_projection_traces": state_ops.merge_context_projection_traces,
        }

        def merge_worker_result(result: dict[str, Any]) -> None:
            for key, reducer in reducer_by_key.items():
                if key in result:
                    local_state[key] = reducer(local_state.get(key, []), result.get(key, []))

        async def execute(task: ResearchTask) -> dict[str, Any]:
            branch = self._branch_state(local_state, task.model_dump(mode="json"))
            branch["current_tasks"] = [
                item.model_dump(mode="json") for item in tasks
            ]
            return await self._worker_node(branch)

        async def commit(
            task: ResearchTask, result: dict[str, Any]
        ) -> FrontierCommit:
            merge_worker_result(result)
            contribution = next(
                (
                    item
                    for item in result.get("contributions", [])
                    if item.get("task_id") == task.task_id
                ),
                {},
            )
            result_roles, result_artifact_ids = artifact_roles_from_result(result)
            produced_roles = list(
                contribution.get("produced_artifact_roles", result_roles)
            )
            produced_ids = list(
                contribution.get("produced_artifact_ids", result_artifact_ids)
            )
            material = bool(
                produced_ids
                or contribution.get("source_ids")
                or contribution.get("evidence_ids")
                or contribution.get("candidate_claim_ids")
                or contribution.get("candidate_lead_ids")
            )
            contribution_status = str(contribution.get("status") or "failed")
            if contribution_status == "completed" and contribution.get("usable"):
                status = "completed"
            elif material:
                status = "partial"
            elif contribution_status in {"failed", "timed_out", "cancelled"}:
                status = "failed"
            else:
                status = "insufficient"
            runtime_update = {
                "task_id": task.task_id,
                "status": status,
                "wave_no": 1,
                "dependency_usable": material or status == "completed",
                "evidence_delta": len(contribution.get("evidence_ids", [])),
                "source_delta": len(contribution.get("source_ids", [])),
                "requirement_support_delta": 0,
                "produced_artifact_roles": produced_roles,
                "produced_artifact_ids": produced_ids,
                "sufficiency_passed": bool(contribution.get("sufficiency_passed")),
                "stop_reason": str(contribution.get("stop_reason") or result.get("error") or ""),
            }
            local_state["task_runtime"] = state_ops.merge_task_runtime(
                local_state.get("task_runtime", []), [runtime_update]
            )
            cell, scope_status = calculate_dual_coverage(local_state)
            local_state["cell_coverage"] = cell.model_dump(mode="json")
            local_state["scope_completeness"] = scope_status.model_dump(mode="json")
            sensor_window_id = (
                f"r{int(state.get('round_no', 1))}-frontier-{task.task_id}"
            )
            sensor_snapshot = progress_snapshot(
                local_state,
                window_id=sensor_window_id,
                enhanced=graph_config.p0_feature_enabled("enhanced_loop_sensor"),
            )
            local_state["sensor_history"] = state_ops.merge_sensor_history(
                local_state.get("sensor_history", []), [sensor_snapshot]
            )
            local_state["sensor_decision"] = evaluate_progress(
                local_state["sensor_history"],
                minimum_windows=max(2, graph_config.min_waves_before_low_gain_stop),
                minimum_stop_signals=graph_config.low_gain_min_signals,
                enhanced=graph_config.p0_feature_enabled("enhanced_loop_sensor"),
            )
            return FrontierCommit(
                runtime_update=runtime_update,
                state_patch=result,
            )

        soft_deadline = float(state.get("started_at", time.time())) + min(
            float(RunBudget.model_validate(state["run_budget"]).max_wall_time_seconds),
            _research_soft_wall_time_seconds(state, self.profiles),
        )

        frontier_result = await run_continuous_frontier(
            tasks=tasks,
            runtime=state.get("task_runtime", []),
            max_concurrency=scope.options.max_workers,
            execute=execute,
            commit=commit,
            priority_order=[
                item.task_id for item in sorted(tasks, key=lambda item: item.priority)
            ],
            may_launch=lambda: time.time() < soft_deadline,
            enhanced=graph_config.p0_feature_enabled("frontier"),
            max_pending_tasks=graph_config.max_plan_tasks,
        )
        local_state["task_runtime"] = frontier_result.runtime
        cell, scope_status = calculate_dual_coverage(local_state)
        window_id = f"r{int(state.get('round_no', 1))}-frontier"
        snapshot = progress_snapshot(
            {
                **local_state,
                "cell_coverage": cell.model_dump(mode="json"),
                "scope_completeness": scope_status.model_dump(mode="json"),
            },
            window_id=window_id,
            enhanced=graph_config.p0_feature_enabled("enhanced_loop_sensor"),
        )
        sensor_history = state_ops.merge_sensor_history(
            local_state.get("sensor_history", []), [snapshot]
        )
        sensor_decision = evaluate_progress(
            sensor_history,
            minimum_windows=max(2, graph_config.min_waves_before_low_gain_stop),
            minimum_stop_signals=graph_config.low_gain_min_signals,
            enhanced=graph_config.p0_feature_enabled("enhanced_loop_sensor"),
        )
        frontier_event = {
            "frontier_event_id": make_id("frontier", state["run_id"], window_id),
            "round_no": int(state.get("round_no", 1)),
            "launch_order": frontier_result.launch_order,
            "completion_order": frontier_result.completion_order,
            "max_running": frontier_result.max_running,
            "task_count": len(frontier_result.tasks),
            "completed_count": len(frontier_result.results),
            "wave_task_ids": {
                str(wave_no): task_ids
                for wave_no, task_ids in sorted(frontier_result.wave_task_ids.items())
            },
        }
        result_by_task = {
            task.task_id: result for task, result in frontier_result.results
        }
        wave_records: list[dict[str, Any]] = []
        for wave_no, task_ids in sorted(frontier_result.wave_task_ids.items()):
            wave_results = [result_by_task[task_id] for task_id in task_ids if task_id in result_by_task]
            source_ids = {
                str(source_id)
                for result in wave_results
                for contribution in result.get("contributions", [])
                for source_id in contribution.get("source_ids", [])
                if source_id
            }
            evidence_ids = {
                str(evidence_id)
                for result in wave_results
                for contribution in result.get("contributions", [])
                for evidence_id in contribution.get("evidence_ids", [])
                if evidence_id
            }
            claim_ids = {
                str(claim_id)
                for result in wave_results
                for contribution in result.get("contributions", [])
                for claim_id in contribution.get("candidate_claim_ids", [])
                if claim_id
            }
            wave_records.append(
                {
                    "wave_id": f"r{int(state.get('round_no', 1))}-w{wave_no}",
                    "round_no": int(state.get("round_no", 1)),
                    "wave_no": wave_no,
                    "task_ids": task_ids,
                    "continuous_frontier": True,
                    "completion_order": [
                        task_id
                        for task_id in frontier_result.completion_order
                        if task_id in set(task_ids)
                    ],
                    "max_running": frontier_result.max_running,
                    "source_delta": len(source_ids),
                    "evidence_delta": len(evidence_ids),
                    "claim_delta": len(claim_ids),
                    "requirement_support_delta": 0,
                    "cell_coverage": cell.model_dump(mode="json"),
                    "scope_completeness": scope_status.model_dump(mode="json"),
                    "sensor_decision": sensor_decision,
                }
            )
        output = {
            key: local_state.get(key, [])
            for key in reducer_by_key
            if key in local_state
        }
        output.update(
            {
                "tasks": [item.model_dump(mode="json") for item in frontier_result.tasks],
                "current_tasks": [
                    item.model_dump(mode="json") for item in frontier_result.tasks
                ],
                "task_runtime": frontier_result.runtime,
                "wave_no": max(frontier_result.wave_task_ids, default=0),
                "current_wave_tasks": [],
                "wave_history": wave_records,
                "frontier_history": [frontier_event],
                "cell_coverage": cell.model_dump(mode="json"),
                "scope_completeness": scope_status.model_dump(mode="json"),
                "sensor_history": [snapshot],
                "sensor_decision": sensor_decision,
            }
        )
        return output

    def _dispatch_wave(self, state: ResearchState):
        wave = state.get("current_wave_tasks", [])
        if not wave:
            return "synthesize_research"
        return [
            Send("research_worker", self._branch_state(state, task))
            for task in wave
        ]

    async def _merge_wave(self, state: ResearchState) -> dict:
        wave_no = int(state.get("wave_no", 0))
        wave_tasks = [
            ResearchTask.model_validate(item)
            for item in state.get("current_wave_tasks", [])
        ]
        contribution_by_task = {
            str(item.get("task_id")): item
            for item in state.get("contributions", [])
            if str(item.get("task_id")) in {task.task_id for task in wave_tasks}
        }
        runtime_updates: list[dict[str, Any]] = []
        runtime_by_task = {
            str(item.get("task_id")): item for item in state.get("task_runtime", [])
        }
        for task in wave_tasks:
            contribution = contribution_by_task.get(task.task_id, {})
            material_counts = (
                len(contribution.get("source_ids", [])),
                len(contribution.get("evidence_ids", [])),
                len(contribution.get("candidate_claim_ids", [])),
                len(contribution.get("candidate_lead_ids", [])),
            )
            has_partial_material = any(material_counts)
            if contribution.get("status") == "completed" and contribution.get("usable"):
                status = "completed"
            elif has_partial_material and contribution.get("status") in {
                "completed",
                "failed",
                "timed_out",
                "cancelled",
            }:
                # Evidence sufficiency is a quality gate, not an execution
                # gate.  Preserve usable partial material for dependent tasks.
                status = "partial"
            elif contribution.get("status") in {"failed", "timed_out", "cancelled"}:
                status = "failed"
            else:
                status = "insufficient"
            runtime_updates.append(
                {
                    "task_id": task.task_id,
                    "status": status,
                    "wave_no": wave_no,
                    "evidence_delta": len(contribution.get("evidence_ids", [])),
                    "source_delta": len(contribution.get("source_ids", [])),
                    "dependency_usable": has_partial_material or status == "completed",
                    "sufficiency_passed": bool(contribution.get("sufficiency_passed")),
                    "requirement_support_delta": len(
                        set(task.produces_requirement_ids)
                        & {
                            str(requirement_id)
                            for claim in state.get("claims", [])
                            if claim.get("claim_id")
                            in set(contribution.get("candidate_claim_ids", []))
                            for requirement_id in claim.get("requirement_ids", [])
                        }
                    ),
                    "produced_artifact_roles": list(
                        contribution.get("produced_artifact_roles", [])
                    ),
                    "produced_artifact_ids": list(
                        contribution.get("produced_artifact_ids", [])
                    ),
                    "stop_reason": str(contribution.get("stop_reason") or "missing_contribution"),
                    "worker_effective_timeout_seconds": runtime_by_task.get(
                        task.task_id, {}
                    ).get("worker_effective_timeout_seconds"),
                    "wave_deadline_at": runtime_by_task.get(task.task_id, {}).get(
                        "wave_deadline_at"
                    ),
                    "worker_tool_deadline_at": runtime_by_task.get(
                        task.task_id, {}
                    ).get("worker_tool_deadline_at"),
                }
            )
        baseline = state.get("wave_baseline", {})
        covered_requirements = {
            str(requirement_id)
            for claim in state.get("claims", [])
            for requirement_id in claim.get("requirement_ids", [])
            if str(requirement_id)
        }
        wave_record = {
            "wave_id": f"r{int(state.get('round_no', 1))}-w{wave_no}",
            "round_no": int(state.get("round_no", 1)),
            "wave_no": wave_no,
            "task_ids": [item.task_id for item in wave_tasks],
            "source_delta": len(state.get("sources", [])) - int(baseline.get("source_count", 0)),
            "evidence_delta": len(state.get("evidence", [])) - int(baseline.get("evidence_count", 0)),
            "claim_delta": len(state.get("claims", [])) - int(baseline.get("claim_count", 0)),
            "requirement_support_delta": len(
                covered_requirements - set(baseline.get("covered_requirement_ids", []))
            ),
            "elapsed_seconds": round(
                max(0.0, time.time() - float(baseline.get("started_at", time.time()))),
                3,
            ),
            "budget": dict(baseline.get("budget", {})),
        }
        runtime_config = getattr(getattr(self, "planner", None), "runtime_config", None)
        graph_config = (
            runtime_config.graph_engineering
            if runtime_config is not None
            else load_runtime_config().graph_engineering
        )
        cell_coverage, scope_completeness = calculate_dual_coverage(state)
        sensor_snapshot = progress_snapshot(
            {
                **state,
                "cell_coverage": cell_coverage.model_dump(mode="json"),
                "scope_completeness": scope_completeness.model_dump(mode="json"),
            },
            window_id=wave_record["wave_id"],
            enhanced=graph_config.p0_feature_enabled("enhanced_loop_sensor"),
        )
        sensor_history = [*state.get("sensor_history", []), sensor_snapshot]
        sensor_decision = evaluate_progress(
            sensor_history,
            minimum_windows=max(
                2,
                graph_config.min_waves_before_low_gain_stop,
            ),
            minimum_stop_signals=graph_config.low_gain_min_signals,
            enhanced=graph_config.p0_feature_enabled("enhanced_loop_sensor"),
        )
        wave_record["cell_coverage"] = cell_coverage.model_dump(mode="json")
        wave_record["scope_completeness"] = scope_completeness.model_dump(mode="json")
        wave_record["sensor_decision"] = sensor_decision
        return {
            "task_runtime": runtime_updates,
            "wave_history": [wave_record],
            "cell_coverage": cell_coverage.model_dump(mode="json"),
            "scope_completeness": scope_completeness.model_dump(mode="json"),
            "sensor_history": [sensor_snapshot],
            "sensor_decision": sensor_decision,
            "events": [
                self._event(
                    state,
                    "wave_merged",
                    145 + (int(state.get("round_no", 1)) - 1) * 200 + wave_no,
                    status="completed",
                    payload=wave_record,
                )
            ],
        }

    async def _deterministic_checks(self, state: ResearchState) -> dict:
        wave_no = int(state.get("wave_no", 0))
        claim_ids = {str(item.get("claim_id")) for item in state.get("claims", [])}
        evidence_ids = {str(item.get("evidence_id")) for item in state.get("evidence", [])}
        dangling_links = [
            str(item.get("link_id"))
            for item in state.get("links", [])
            if str(item.get("claim_id")) not in claim_ids
            or str(item.get("evidence_id")) not in evidence_ids
        ]
        wave_id = f"r{int(state.get('round_no', 1))}-w{int(state.get('wave_no', 0))}"
        current = next(
            (dict(item) for item in state.get("wave_history", []) if item.get("wave_id") == wave_id),
            {"wave_id": wave_id},
        )
        current["deterministic_checks"] = {
            "dangling_link_ids": dangling_links,
            "passed": not dangling_links,
        }
        evidence_by_id = {
            str(item.get("evidence_id")): item for item in state.get("evidence", [])
        }
        valid_links = [
            item
            for item in state.get("links", [])
            if item.get("relation") == "supports"
            and item.get("entailment_status") == "verified"
            and str(item.get("claim_id")) in claim_ids
            and str(item.get("evidence_id")) in evidence_ids
        ]
        linked_claims_by_requirement: dict[str, set[str]] = {}
        source_ids_by_requirement: dict[str, set[str]] = {}
        for link in valid_links:
            claim = next(
                (item for item in state.get("claims", []) if item.get("claim_id") == link.get("claim_id")),
                {},
            )
            evidence = evidence_by_id.get(str(link.get("evidence_id")), {})
            requirements = {
                str(item)
                for item in [*claim.get("requirement_ids", []), *evidence.get("requirement_ids", [])]
                if str(item)
            }
            for requirement_id in requirements:
                linked_claims_by_requirement.setdefault(requirement_id, set()).add(
                    str(link.get("claim_id"))
                )
                if evidence.get("source_id"):
                    source_ids_by_requirement.setdefault(requirement_id, set()).add(
                        str(evidence.get("source_id"))
                    )
        requirement_definitions = {
            str(item.get("requirement_id")): item
            for item in state.get("current_plan", {}).get("requirements", [])
            if isinstance(item, dict) and item.get("requirement_id")
        }
        requirement_ids = set(requirement_definitions)
        for task in state.get("current_tasks", []):
            requirement_ids.update(str(item) for item in task.get("requirement_ids", []) if str(item))
            requirement_ids.update(str(item) for item in task.get("produces_requirement_ids", []) if str(item))

        # A verified claim is useful evidence, but it does not prove that the
        # whole delegated requirement is complete.  Only a producer task that
        # finished and passed its own sufficiency floor can promote linked
        # evidence from partial to sufficient.  This also prevents a broad or
        # repeated-field task from being superseded after one incidental claim.
        producer_task_ids_by_requirement: dict[str, set[str]] = {}
        for task in state.get("current_tasks", []):
            task_id = str(task.get("task_id") or "")
            if not task_id:
                continue
            produced = task.get("produces_requirement_ids", []) or task.get(
                "requirement_ids", []
            )
            for requirement_id in produced:
                if str(requirement_id):
                    producer_task_ids_by_requirement.setdefault(
                        str(requirement_id), set()
                    ).add(task_id)
        completed_task_ids = {
            str(item.get("task_id"))
            for item in state.get("task_runtime", [])
            if item.get("status") == "completed" and item.get("task_id")
        }
        sufficient_contribution_task_ids = {
            str(item.get("task_id"))
            for item in state.get("contributions", [])
            if item.get("task_id")
            and item.get("status") == "completed"
            and bool(item.get("sufficiency_passed") or item.get("usable"))
        }
        completed_sufficient_task_ids = (
            completed_task_ids & sufficient_contribution_task_ids
        )
        requirement_status = []
        for requirement_id in sorted(requirement_ids):
            definition = requirement_definitions.get(requirement_id, {})
            required_elements = [
                str(value).strip()
                for value in definition.get("required_elements", [])
                if str(value).strip()
            ]
            linked_claim_ids = sorted(linked_claims_by_requirement.get(requirement_id, set()))
            source_ids = sorted(source_ids_by_requirement.get(requirement_id, set()))
            producer_task_ids = producer_task_ids_by_requirement.get(
                requirement_id, set()
            )
            completed_producer_ids = sorted(
                producer_task_ids & completed_sufficient_task_ids
            )
            all_producers_passed = bool(producer_task_ids) and (
                producer_task_ids <= completed_sufficient_task_ids
            )
            has_requirement_material = bool(linked_claim_ids) or any(
                requirement_id
                in {str(value) for value in item.get("requirement_ids", [])}
                for item in state.get("claims", []) + state.get("evidence", [])
            )
            operational_status = (
                "sufficient"
                if linked_claim_ids and all_producers_passed
                else "partial"
                if has_requirement_material
                else "missing"
            )
            # Producer completion and semantic element coverage are different
            # signals. When a composite requirement has not been assessed
            # element by element, expose a conservative public status while
            # retaining the operational result for scheduling compatibility.
            status = (
                "partial"
                if required_elements and operational_status == "sufficient"
                else operational_status
            )
            requirement_status.append(
                {
                    "requirement_id": requirement_id,
                    "text": str(definition.get("text") or ""),
                    "kind": str(definition.get("kind") or "fact"),
                    "importance": str(
                        definition.get("importance") or "required"
                    ),
                    "target_section": str(
                        definition.get("target_section") or ""
                    ),
                    "item_key": str(definition.get("item_key") or ""),
                    "field_key": str(definition.get("field_key") or ""),
                    "case_id": str(definition.get("case_id") or ""),
                    "required_elements": required_elements,
                    "required_element_statuses": [
                        {
                            "element": element,
                            "status": "unassessed",
                            "reason": (
                                "Deterministic coverage does not infer semantic "
                                "completion from requirement-level claim links."
                            ),
                        }
                        for element in required_elements
                    ],
                    "status": status,
                    "operational_status": operational_status,
                    "semantic_status": (
                        "unassessed" if required_elements else "not_applicable"
                    ),
                    "coverage_authority": "advisory",
                    "semantic_completeness_assessed": False,
                    # Keep the two signals separate.  A Worker can finish its
                    # delegated sufficiency contract while only part of a
                    # repeated/table requirement is evidenced; downstream
                    # LLMs must see that distinction instead of treating task
                    # completion as field coverage.
                    "evidence_coverage_status": (
                        "linked" if linked_claim_ids else "missing"
                    ),
                    "task_completion_status": (
                        "sufficient"
                        if all_producers_passed
                        else "pending"
                    ),
                    "task_sufficiency_is_not_requirement_coverage": True,
                    "supported_claim_ids": linked_claim_ids,
                    "producer_task_ids": sorted(producer_task_ids),
                    "completed_sufficient_producer_task_ids": completed_producer_ids,
                    "all_producers_passed_sufficiency": all_producers_passed,
                    "effective_source_ids": source_ids,
                    "independent_domains": sorted(
                        {
                            independent_domain(str(source.get("canonical_uri") or ""))
                            for source in state.get("sources", [])
                            if str(source.get("source_id")) in set(source_ids)
                        }
                        - {""}
                    ),
                    "last_changed_wave": wave_no,
                    "assessment_reason": (
                        "verified claim-evidence link and a producing task passed sufficiency; "
                        "semantic element completeness remains for the Synthesizer"
                        if operational_status == "sufficient"
                        else "verified material exists but no producing task has passed sufficiency"
                        if operational_status == "partial" and linked_claim_ids
                        else "claim or evidence exists without a verified link"
                        if operational_status == "partial"
                        else "no linked evidence"
                    ),
                }
            )
        blueprint = state.get("current_plan", {}).get("coverage_blueprint", {})
        case_rows = (
            blueprint.get("candidate_cases", [])
            if isinstance(blueprint, dict)
            else []
        )
        repeated_fields = {
            str(item).strip().casefold()
            for item in (blueprint.get("repeated_case_fields", []) if isinstance(blueprint, dict) else [])
            if str(item).strip()
        }
        evidence_coverage_by_id = {
            str(item.get("requirement_id")): str(
                item.get("evidence_coverage_status") or "missing"
            )
            for item in requirement_status
            if item.get("requirement_id")
        }

        def compact(value: Any) -> str:
            return "".join(char for char in str(value).casefold() if char.isalnum())

        known_case_ids: set[str] = set()
        incomplete_cell_ids: set[str] = set()
        covered_cell_count = 0
        expected_cell_count = 0
        fully_covered_case_count = 0
        for case in case_rows if isinstance(case_rows, list) else []:
            if not isinstance(case, dict) or not case.get("case_id"):
                continue
            case_id = str(case.get("case_id"))
            known_case_ids.add(case_id)
            field_ids = case.get("field_requirement_ids", {})
            fields = list(field_ids) if isinstance(field_ids, dict) else []
            if not fields:
                fields = [str(item) for item in case.get("fields", []) if str(item).strip()]
            case_complete = bool(fields)
            for field in fields:
                requirement_id = str(field_ids.get(field) or "") if isinstance(field_ids, dict) else ""
                if not requirement_id:
                    continue
                expected_cell_count += 1
                if evidence_coverage_by_id.get(requirement_id) == "linked":
                    covered_cell_count += 1
                else:
                    case_complete = False
                    incomplete_cell_ids.add(requirement_id)
            if case_complete and fields:
                fully_covered_case_count += 1

        unverified_leads = [
            item
            for item in state.get("candidate_leads", [])
            if isinstance(item, dict)
            and str(item.get("status") or "unverified") == "unverified"
            and (item.get("title") or item.get("identifier"))
        ]
        unmatched_lead_ids: list[str] = []
        case_signatures = {
            signature
            for case in case_rows if isinstance(case_rows, list) and isinstance(case, dict)
            for signature in [
                compact(case.get("label")),
                *[compact(item) for item in case.get("aliases", [])],
                *[compact(item) for item in case.get("identifiers", [])],
            ]
            if signature
        }
        for lead in unverified_leads:
            signatures = [compact(lead.get("title")), compact(lead.get("identifier"))]
            if not any(signature and signature in case_signatures for signature in signatures):
                unmatched_lead_ids.append(str(lead.get("lead_id") or ""))

        collection_active = bool(repeated_fields or expected_cell_count or case_rows)
        collection_inventory_open = bool(incomplete_cell_ids or unmatched_lead_ids)
        collection_task_ids: list[str] = []
        if collection_active and collection_inventory_open:
            task_rows = state.get("current_tasks", [])
            ready_task_rows, _ = ready_tasks(
                [ResearchTask.model_validate(item) for item in task_rows],
                state.get("task_runtime", []),
            )
            runnable_ids = {
                str(item.get("task_id"))
                for item in state.get("task_runtime", [])
                if item.get("task_id")
                and item.get("status") == "running"
            }
            runnable_ids.update(item.task_id for item in ready_task_rows)
            lead_by_id = {str(item.get("lead_id")): item for item in unverified_leads}
            for task in task_rows:
                if str(task.get("task_id")) not in runnable_ids:
                    continue
                refs = {
                    str(value)
                    for key in ("requirement_ids", "produces_requirement_ids", "consumes_requirement_ids")
                    for value in task.get(key, [])
                    if str(value)
                }
                depends = {str(value) for value in task.get("depends_on", []) if str(value)}
                if refs & incomplete_cell_ids or any(
                    str(lead_by_id[lead_id].get("task_id")) in depends
                    or refs & {str(value) for value in lead_by_id[lead_id].get("requirement_ids", [])}
                    for lead_id in unmatched_lead_ids
                    if lead_id in lead_by_id
                ):
                    collection_task_ids.append(str(task.get("task_id")))

        collection_coverage = {
            "active": collection_active,
            "known_case_count": len(known_case_ids),
            "verified_case_count": fully_covered_case_count,
            "expected_cell_count": expected_cell_count,
            "covered_cell_count": covered_cell_count,
            "case_coverage_ratio": round(
                fully_covered_case_count / len(known_case_ids)
                if known_case_ids else 0.0,
                4,
            ),
            "cell_coverage_ratio": round(
                covered_cell_count / expected_cell_count
                if expected_cell_count else 0.0,
                4,
            ),
            "inventory_open": collection_inventory_open,
            "unverified_lead_count": len(unmatched_lead_ids),
            "incomplete_cell_requirement_ids": sorted(incomplete_cell_ids),
            "actionable_task_ids": sorted(set(collection_task_ids)),
        }
        return {
            "wave_history": [current],
            "requirement_status": requirement_status,
            "collection_coverage": collection_coverage,
        }

    async def _control_wave(self, state: ResearchState) -> dict:
        graph_config = self.planner.runtime_config.graph_engineering
        current_round = int(state.get("round_no", 1))
        round_waves = [
            item
            for item in state.get("wave_history", [])
            if int(item.get("round_no", 0)) == current_round
        ]
        zero_gain_waves = 0
        for item in sorted(round_waves, key=lambda value: int(value.get("wave_no", 0)), reverse=True):
            if any(int(item.get(field, 0)) > 0 for field in ("source_delta", "evidence_delta", "claim_delta")):
                break
            zero_gain_waves += 1
        tasks = [ResearchTask.model_validate(item) for item in state.get("current_tasks", [])]
        runtime = list(state.get("task_runtime", []))
        ready, blocked = ready_tasks(tasks, runtime)
        pending = [
            item
            for item in runtime
            if item.get("status") == "pending"
            and item.get("task_id") in {task.task_id for task in tasks}
        ]
        required_status = [
            item
            for item in state.get("requirement_status", [])
            if str(item.get("importance") or "required") == "required"
        ]
        covered_required = [
            item for item in required_status if item.get("status") != "missing"
        ]
        required_coverage_ratio = (
            len(covered_required) / len(required_status) if required_status else 0.0
        )
        collection_coverage = state.get("collection_coverage", {})
        collection_actionable_task_ids = {
            str(item)
            for item in collection_coverage.get("actionable_task_ids", [])
            if str(item)
        }
        collection_incomplete_actionable = bool(
            collection_coverage.get("active")
            and collection_coverage.get("inventory_open")
            and collection_actionable_task_ids
        )
        low_gain_requirement_ceiling = max(
            1,
            math.ceil(
                len(required_status)
                * float(graph_config.low_gain_requirement_fraction)
            ),
        )
        low_gain_waves = 0
        for item in sorted(
            round_waves,
            key=lambda value: int(value.get("wave_no", 0)),
            reverse=True,
        ):
            if int(item.get("requirement_support_delta", 0)) > low_gain_requirement_ceiling:
                break
            low_gain_waves += 1

        missing_required_ids = {
            str(item.get("requirement_id"))
            for item in required_status
            if item.get("status") == "missing" and item.get("requirement_id")
        }
        missing_required_ids.update(
            str(item)
            for item in collection_coverage.get("incomplete_cell_requirement_ids", [])
            if str(item)
        )
        protected_high_value_tasks = [
            task.task_id
            for task in ready
            if float(task.expected_value) >= 0.8
            and missing_required_ids
            & {
                str(item)
                for item in (
                    task.produces_requirement_ids or task.requirement_ids
                )
            }
            or (
                float(task.expected_value) >= 0.8
                and task.task_id in collection_actionable_task_ids
            )
        ]
        low_gain_stop_signals = {
            "minimum_waves_reached": (
                len(round_waves) >= graph_config.min_waves_before_low_gain_stop
            ),
            "coverage_saturated": (
                required_coverage_ratio >= graph_config.low_gain_coverage_ratio
                and not collection_incomplete_actionable
            ),
            "consecutive_low_gain": (
                low_gain_waves >= graph_config.max_low_gain_waves
            ),
            "no_high_value_missing_task": not protected_high_value_tasks,
        }
        low_gain_signal_count = sum(low_gain_stop_signals.values())
        low_gain_value_signal_count = sum(
            value
            for key, value in low_gain_stop_signals.items()
            if key != "minimum_waves_reached"
        )
        required_value_signals = max(1, graph_config.low_gain_min_signals - 1)
        low_gain_limit_reached = (
            bool(required_status)
            and low_gain_stop_signals["minimum_waves_reached"]
            and low_gain_value_signal_count >= required_value_signals
        )
        should_continue = bool(ready or pending) and (
            zero_gain_waves < graph_config.max_zero_gain_waves
        ) and not low_gain_limit_reached
        route = "schedule_wave" if should_continue else "synthesize_research"
        decision = {
            "wave_id": f"r{current_round}-w{int(state.get('wave_no', 0))}",
            "round_no": current_round,
            "wave_no": int(state.get("wave_no", 0)),
            "decision": "continue" if should_continue else "synthesize",
            "route": route,
            "ready_task_ids": [item.task_id for item in ready],
            "pending_task_ids": [str(item.get("task_id")) for item in pending],
            "blocked_task_ids": [str(item.get("task_id")) for item in blocked],
            "zero_gain_waves": zero_gain_waves,
            "low_gain_waves": low_gain_waves,
            "low_gain_requirement_ceiling": low_gain_requirement_ceiling,
            "required_coverage_ratio": round(required_coverage_ratio, 4),
            "protected_high_value_task_ids": protected_high_value_tasks,
            "collection_coverage": collection_coverage,
            "low_gain_stop_signals": low_gain_stop_signals,
            "low_gain_signal_count": low_gain_signal_count,
            "low_gain_value_signal_count": low_gain_value_signal_count,
            "low_gain_required_value_signals": required_value_signals,
            "low_gain_min_signals": graph_config.low_gain_min_signals,
            "reason": (
                "dependency_ready_tasks_remain"
                if should_continue
                else "low_gain_limit"
                if low_gain_limit_reached
                else "zero_gain_limit"
                if zero_gain_waves >= graph_config.max_zero_gain_waves
                else "dag_terminal"
            ),
        }
        return {
            "controller_decision": decision,
            "controller_history": [decision],
            "events": [
                self._event(
                    state,
                    "wave_controlled",
                    155 + (current_round - 1) * 200 + int(state.get("wave_no", 0)),
                    status=decision["decision"],
                    payload=decision,
                )
            ],
        }

    @staticmethod
    def _route_controller(state: ResearchState) -> str:
        return str(state.get("controller_decision", {}).get("route") or "synthesize_research")

    @staticmethod
    def _dependency_context(state: ResearchState, task: dict) -> dict[str, Any]:
        dependency_ids = {str(item) for item in task.get("depends_on", [])}
        if not dependency_ids:
            return {"dependencies": [], "claims": [], "evidence": [], "sources": [], "candidate_leads": []}
        runtime_by_task = {
            str(item.get("task_id")): item
            for item in state.get("task_runtime", [])
            if str(item.get("task_id")) in dependency_ids
        }
        contributions = {
            str(item.get("task_id")): item
            for item in state.get("contributions", [])
            if str(item.get("task_id")) in dependency_ids
            and bool(
                runtime_by_task.get(str(item.get("task_id")), {}).get(
                    "dependency_usable"
                )
            )
        }
        claim_ids = {
            str(claim_id)
            for item in contributions.values()
            for claim_id in item.get("candidate_claim_ids", [])
        }
        evidence_ids = {
            str(evidence_id)
            for item in contributions.values()
            for evidence_id in item.get("evidence_ids", [])
        }
        source_ids = {
            str(source_id)
            for item in contributions.values()
            for source_id in item.get("source_ids", [])
        }
        return {
            "dependencies": [
                {
                    "task_id": task_id,
                    "result_brief": next(
                        (item.get("result_brief", "") for item in contributions.values() if item.get("task_id") == task_id),
                        "",
                    ),
                    "status": runtime_by_task.get(task_id, {}).get(
                        "status", "unavailable"
                    ),
                }
                for task_id in sorted(dependency_ids)
            ],
            "claims": [
                {
                    "claim_id": item.get("claim_id"),
                    "text": item.get("text"),
                    "requirement_ids": item.get("requirement_ids", []),
                }
                for item in state.get("claims", [])
                if str(item.get("claim_id")) in claim_ids
            ],
            "evidence": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "source_id": item.get("source_id"),
                    "quote": item.get("quote"),
                    "requirement_ids": item.get("requirement_ids", []),
                }
                for item in state.get("evidence", [])
                if str(item.get("evidence_id")) in evidence_ids
            ],
            "sources": [
                {
                    "source_id": item.get("source_id"),
                    "title": item.get("title"),
                    "canonical_uri": item.get("canonical_uri"),
                    "quality_tier": item.get("quality_tier"),
                }
                for item in state.get("sources", [])
                if str(item.get("source_id")) in source_ids
            ],
            # Candidate leads are intentionally labelled navigation-only. A
            # downstream Worker must verify them before they can affect claims.
            "candidate_leads": [
                item
                for item in state.get("candidate_leads", [])
                if str(item.get("task_id")) in dependency_ids
                and item.get("status") == "unverified"
            ],
        }

    def _branch_state(self, state: ResearchState, task: dict) -> dict:
        completion_contract = str(task.get("completion_contract") or "discover_sources")
        projection_role = (
            "explore_worker"
            if completion_contract in {"discover_sources", "extract_entities", "audit_scope"}
            else "verify_worker"
            if completion_contract in {"verify_entity", "resolve_conflict"}
            else "search_worker"
        )
        role_projection = project_context(
            role=projection_role,
            state=state,
            task=task,
        )
        dependency_context = self._dependency_context(state, task)
        dependency_context["role_projection"] = role_projection
        dependency_context["progress_sensor"] = state.get("sensor_decision", {})
        branch = {
            "run_id": state["run_id"],
            "user_id": state.get("user_id", "local-user"),
            "project_id": state.get("project_id", "deepresearch-default"),
            "session_id": state.get("session_id", ""),
            "turn_id": state.get("turn_id", ""),
            "request": state["request"],
            "scope": state["scope"],
            "run_budget": state["run_budget"],
            "started_at": state.get("started_at", time.time()),
            "current_task": task,
            "current_tasks": state.get("current_tasks", []),
            "current_plan": state.get("current_plan", {}),
            "plan_history": state.get("plan_history", []),
            "round_no": state.get("round_no", 1),
            "unresolved_gaps": state.get("unresolved_gaps", []),
            "usage_events": state.get("usage_events", []),
            "planner_memory": state.get("planner_memory", {}),
            "dependency_context": dependency_context,
            "wave_budget": state.get("wave_budget", {}),
            "wave_deadline_at": state.get("wave_deadline_at"),
            "wave_tool_deadline_at": state.get("wave_tool_deadline_at"),
        }
        planner_memory = MemoryPacket.model_validate(
            state.get("planner_memory")
            or {"role": "planner"}
        )
        branch["worker_memory"] = LayeredMemory.worker_packet(
            planner_memory,
            ResearchTask.model_validate(task),
            state,
            budget_tokens=self.planner.runtime_config.memory.worker_memory_budget_tokens,
            sentence_limit=self.planner.runtime_config.memory.projection_sentence_limit,
        ).model_dump(mode="json")
        projected = branch.get("dependency_context", {})
        branch["context_projection_traces"] = [
            {
                **projection_trace(
                    run_id=state["run_id"],
                    role=projection_role,
                    task_id=str(task.get("task_id") or ""),
                    projection=role_projection,
                    round_no=int(task.get("round_no", state.get("round_no", 1))),
                ),
                "dependency_task_ids": list(task.get("depends_on", [])),
                "claim_ids": [item.get("claim_id") for item in projected.get("claims", [])],
                "evidence_ids": [item.get("evidence_id") for item in projected.get("evidence", [])],
                "candidate_lead_ids": [item.get("lead_id") for item in projected.get("candidate_leads", [])],
                "source_snapshot_mutated": False,
            }
        ]
        for key in (
            "sources",
            "evidence",
            "claims",
            "source_leads",
            "entity_candidates",
            "evidence_bindings",
        ):
            branch[key] = role_projection.get(key, [])
        branch["failure_memory"] = role_projection.get("failure_guidance", [])
        source_ids = {
            str(item.get("source_id"))
            for item in branch.get("sources", [])
            if item.get("source_id")
        }
        claim_ids = {
            str(item.get("claim_id"))
            for item in branch.get("claims", [])
            if item.get("claim_id")
        }
        evidence_ids = {
            str(item.get("evidence_id"))
            for item in branch.get("evidence", [])
            if item.get("evidence_id")
        }
        branch["source_snapshots"] = [
            item
            for item in state.get("source_snapshots", [])
            if str(item.get("source_id")) in source_ids
        ]
        branch["links"] = [
            item
            for item in state.get("links", [])
            if str(item.get("claim_id")) in claim_ids
            or str(item.get("evidence_id")) in evidence_ids
        ]
        relevant_task_ids = {
            str(task.get("task_id") or ""),
            *{str(item) for item in task.get("depends_on", [])},
        }
        for key in ("search_traces", "fetch_traces"):
            branch[key] = [
                item
                for item in state.get(key, [])
                if str(item.get("task_id") or "") in relevant_task_ids
            ]
        for key in ("tasks", "contributions"):
            branch[key] = [
                item
                for item in state.get(key, [])
                if str(item.get("task_id") or "") in relevant_task_ids
            ]
        for key in (
            "search_schemas",
            "schema_revisions",
            "evidence_edges",
            "coverage_changes",
            "observation_commits",
        ):
            branch[key] = state.get(key, [])
        branch["cell_coverage"] = state.get("cell_coverage", {})
        branch["scope_completeness"] = state.get("scope_completeness", {})
        return branch

    def _dispatch_tasks(self, state: ResearchState):
        return [
            Send("research_worker", self._branch_state(state, task))
            for task in state.get("current_tasks", [])
        ]

    async def _worker_node(self, state: ResearchState) -> dict:
        task = ResearchTask.model_validate(state["current_task"])
        reporter = current_reporter()
        workers = list(state.get("current_tasks", []))
        worker_index = next(
            (
                index
                for index, item in enumerate(workers, 1)
                if item.get("task_id") == task.task_id
            ),
            task.priority,
        )
        worker_id = f"R{task.round_no}-W{worker_index}"
        operation_id = f"worker:{task.task_id}"
        worker_metrics = (
            reporter.mark_worker(task.round_no, task.task_id, "running")
            if reporter is not None
            else {}
        )
        emit_progress(
            "worker_started",
            phase="worker",
            agent_role="worker",
            status="started",
            message=f"{worker_id} 开始研究：{task.research_dimension}",
            operation_id=operation_id,
            round_no=task.round_no,
            task_id=task.task_id,
            worker_id=worker_id,
            visibility="normal",
            metrics={**worker_metrics, "objective": task.objective},
        )
        run_budget = RunBudget.model_validate(state["run_budget"])
        worker_started_at = time.time()
        remaining_run_time = run_budget.max_wall_time_seconds - (
            worker_started_at - state.get("started_at", worker_started_at)
        )
        graph_config = self.planner.runtime_config.graph_engineering
        wave_deadline_at = float(
            state.get("wave_deadline_at")
            or (worker_started_at + max(0.0, remaining_run_time))
        )
        remaining_wave_time = max(0.0, wave_deadline_at - worker_started_at)
        timeout = min(
            float(task.budget.max_wall_time_seconds),
            float(graph_config.max_worker_wall_time_seconds),
            max(0.0, remaining_run_time),
            remaining_wave_time,
        )
        worker_deadline_at = worker_started_at + timeout
        finalization_reserve = min(
            float(graph_config.worker_finalization_reserve_seconds),
            timeout * 0.20,
        )
        configured_tool_deadline = state.get("wave_tool_deadline_at")
        worker_tool_deadline_at = min(
            worker_deadline_at - finalization_reserve,
            float(configured_tool_deadline)
            if configured_tool_deadline is not None
            else worker_deadline_at,
        )
        worker_reflection_deadline_at = max(
            worker_started_at,
            worker_deadline_at
            - min(float(graph_config.worker_cleanup_reserve_seconds), timeout * 0.05),
        )
        worker_state = dict(state)
        worker_state.update(
            {
                "worker_deadline_at": worker_deadline_at,
                "worker_tool_deadline_at": worker_tool_deadline_at,
                "worker_reflection_deadline_at": worker_reflection_deadline_at,
            }
        )
        try:
            try:
                if timeout <= 0:
                    raise TimeoutError("Run wall-time budget exhausted")
                result = await asyncio.wait_for(
                    self.worker.run(worker_state, task), timeout=timeout
                )
            except TimeoutError:
                contribution = ResearchContribution(
                    task_id=task.task_id,
                    agent_id="worker",
                    model_id=self.worker.llm.model_id,
                    attempt=task.attempt,
                    status="timed_out",
                    stop_reason="budget" if remaining_run_time <= 0 else "timeout",
                    usable=False,
                    result_brief="",
                    knowledge_gaps=["Research worker timed out"],
                    started_at=utc_now(),
                    ended_at=utc_now(),
                    error=f"Timeout after {timeout} seconds",
                )
                result = {"contributions": [contribution.model_dump()]}
        finally:
            await asyncio.gather(
                self.worker.fetch_budget_pool.complete(
                    state["run_id"], task.round_no, task.task_id
                ),
                self.worker.search_budget_pool.complete(
                    state["run_id"], task.round_no, task.task_id
                ),
            )
        if graph_config.atomic_observation_commits_enabled:
            delta = delta_from_worker_result(
                run_id=state["run_id"],
                task=task,
                result=result,
                existing_failure_records=worker_state.get("failure_memory", []),
                enhanced_failure_memory=graph_config.p0_feature_enabled(
                    "failure_memory"
                ),
            )
            rejection: dict[str, Any] | None = None
            delta, pruning = prune_invalid_observation_items(worker_state, delta)
            rejected_evidence = pruning["rejected_evidence"]
            rejected_claim_ids = pruning["rejected_claim_ids"]
            if rejected_evidence or rejected_claim_ids:
                rejection = {
                    "rejection_id": make_id(
                        "observation-rejection",
                        delta.delta_id,
                        rejected_evidence,
                        rejected_claim_ids,
                    ),
                    "delta_id": delta.delta_id,
                    "task_id": task.task_id,
                    "reason": "invalid_observation_items_pruned",
                    "action": "invalid_items_pruned_valid_items_preserved",
                    **pruning,
                }
            try:
                committed = commit_observation_delta(
                    worker_state,
                    delta,
                    append_only_evidence=graph_config.p0_feature_enabled(
                        "append_only_evidence"
                    ),
                )
            except ObservationValidationError as exc:
                rejection = {
                    "rejection_id": make_id(
                        "observation-rejection", delta.delta_id, str(exc)
                    ),
                    "delta_id": delta.delta_id,
                    "task_id": task.task_id,
                    "reason": str(exc),
                    "action": "unexpected_structural_validation_failure",
                }
                safe_delta = delta.model_copy(
                    update={
                        "source_leads": [],
                        "entity_candidates": [],
                        "evidence_records": [],
                        "claims": [],
                        "links": [],
                        "bindings": [],
                        "graph_edges": [],
                        "coverage_changes": [],
                    }
                )
                committed = commit_observation_delta(
                    worker_state,
                    safe_delta,
                    append_only_evidence=graph_config.p0_feature_enabled(
                        "append_only_evidence"
                    ),
                )
            result.update(committed["state_patch"])
            if rejection is not None:
                result["observation_rejections"] = [rejection]
        result["events"] = [
            self._event(
                state,
                "worker_completed",
                100 + (task.round_no - 1) * 200 + task.priority * 10,
                task_id=task.task_id,
                payload={
                    "round_no": task.round_no,
                    "sufficiency": result.get("contributions", [{}])[0].get("sufficiency_passed", False),
                    "successful_extractions": result.get("contributions", [{}])[0].get("successful_extractions", 0),
                    "evidence_count": result.get("contributions", [{}])[0].get("evidence_count", 0),
                    "worker_effective_timeout_seconds": round(timeout, 3),
                    "worker_deadline_at": worker_deadline_at,
                    "worker_tool_deadline_at": worker_tool_deadline_at,
                },
            )
        ]
        result["context_projection_traces"] = state.get(
            "context_projection_traces", []
        )
        contribution = result.get("contributions", [{}])[0]
        if contribution.get("status") == "completed" and contribution.get("usable"):
            worker_status = "completed"
        elif contribution.get("status") == "timed_out":
            worker_status = "timed_out"
        elif contribution.get("error"):
            worker_status = "failed"
        else:
            worker_status = "insufficient"
        worker_metrics = (
            reporter.mark_worker(task.round_no, task.task_id, worker_status)
            if reporter is not None
            else {}
        )
        emit_progress(
            "worker_completed",
            phase="worker",
            agent_role="worker",
            status="completed" if worker_status == "completed" else "failed",
            message=(
                f"{worker_id} 完成：{contribution.get('effective_source_count', 0)} 来源、"
                f"{contribution.get('evidence_count', 0)} 证据"
                if worker_status == "completed"
                else f"{worker_id} 结束：{worker_status}"
            ),
            operation_id=operation_id,
            round_no=task.round_no,
            task_id=task.task_id,
            worker_id=worker_id,
            visibility="normal",
            metrics={
                **worker_metrics,
                "worker_status": worker_status,
                "sources": contribution.get("effective_source_count", 0),
                "evidence": contribution.get("evidence_count", 0),
                "claims": len(contribution.get("candidate_claim_ids", [])),
                "sufficiency_passed": contribution.get("sufficiency_passed", False),
                "stop_reason": contribution.get("stop_reason"),
            },
        )
        return result

    async def _synthesize_research(self, state: ResearchState) -> dict:
        scope = ResearchScope.model_validate(state["scope"])
        run_budget = RunBudget.model_validate(state["run_budget"])
        round_no = int(state.get("round_no", 1))
        operation_id = f"synthesizer:r{round_no}"
        emit_progress(
            "synthesizer_started",
            phase="synthesis",
            agent_role="synthesizer",
            status="started",
            message=f"Synthesizer 正在评估第 {round_no} 轮材料",
            operation_id=operation_id,
            round_no=round_no,
            visibility="quiet",
        )
        remaining = remaining_task_budget(
            run_budget,
            state.get("usage_events", []),
            round_no=round_no + 1,
            max_rounds=scope.options.max_rounds,
        )
        plan = state.get("current_plan", {})
        evidence_delta = len(state.get("claims", [])) - int(plan.get("starting_claim_count", 0))
        source_delta = len(state.get("sources", [])) - int(plan.get("starting_source_count", 0))
        current_independent_domains = {
            independent_domain(str(item.get("canonical_uri") or ""))
            for item in state.get("sources", [])
            if item.get("canonical_uri")
        } - {""}
        starting_independent_domains = set(plan.get("starting_independent_domains", []))
        independent_source_delta = len(
            current_independent_domains - starting_independent_domains
        )
        starting_high_priority_gap_count = int(
            plan.get("starting_high_priority_gap_count", 0)
        )
        starting_covered_requirement_ids = set(
            plan.get("starting_covered_requirement_ids", [])
        )
        current_covered_requirement_ids = {
            str(requirement_id)
            for claim in state.get("claims", [])
            for requirement_id in claim.get("requirement_ids", [])
            if str(requirement_id)
        }
        requirement_coverage_delta = len(
            current_covered_requirement_ids - starting_covered_requirement_ids
        )
        current_task_ids = {
            item.get("task_id") for item in state.get("current_tasks", [])
        }
        contribution_progress = any(
            item.get("task_id") in current_task_ids
            and (item.get("evidence_ids") or item.get("source_ids"))
            for item in state.get("contributions", [])
        )
        now_timestamp = time.time()
        elapsed_run_time = now_timestamp - state.get("started_at", now_timestamp)
        hard_remaining_run_time = max(
            0.0, run_budget.max_wall_time_seconds - elapsed_run_time
        )
        soft_wall_time = _research_soft_wall_time_seconds(state, self.profiles)
        remaining_run_time = max(0.0, soft_wall_time - elapsed_run_time)
        round_elapsed_time = max(
            0.001,
            now_timestamp - float(plan.get("round_started_at", now_timestamp)),
        )
        # Gap-closing rounds are intentionally narrower than the breadth-first
        # first round. Requiring enough time to repeat the whole prior round
        # forced a partial answer even when several minutes remained for a
        # focused re-plan.
        required_next_round_seconds = _required_next_round_seconds(
            round_elapsed_time,
            soft_wall_time,
            scope.options.max_rounds,
        )
        has_full_round_time = remaining_run_time >= required_next_round_seconds
        can_continue = (
            round_no < scope.options.max_rounds
            and remaining.max_fetches > 0
            and (
                remaining.max_searches > 0
                or _has_unattempted_search_candidates(state)
            )
            and has_full_round_time
            and (
                round_no == 1
                or evidence_delta > 0
                or source_delta > 0
                or contribution_progress
            )
        )
        round_progress = {
            "starting_high_priority_gap_count": starting_high_priority_gap_count,
            "starting_independent_source_count": len(starting_independent_domains),
            "current_independent_source_count": len(current_independent_domains),
            "independent_source_delta": independent_source_delta,
            "evidence_delta": evidence_delta,
            "source_delta": source_delta,
            "requirement_coverage_delta": requirement_coverage_delta,
            "round_elapsed_seconds": round(round_elapsed_time, 3),
            "remaining_run_seconds": round(remaining_run_time, 3),
            "required_next_round_seconds": round(required_next_round_seconds, 3),
            "hard_remaining_run_seconds": round(hard_remaining_run_time, 3),
            "research_soft_wall_time_seconds": round(soft_wall_time, 3),
            "full_round_time_reserved": has_full_round_time,
        }
        decision, response, trace = await self.synthesizer.assess(
            state,
            can_continue=can_continue,
            round_progress=round_progress,
        )
        current_high_priority_gap_count = sum(
            1 for item in decision.missing_information if item.priority == "high"
        )
        high_priority_gaps_reduced = (
            current_high_priority_gap_count < starting_high_priority_gap_count
        )
        requirement_matrix = self.profiles.get(scope.profile_id).runtime_policy.requirement_matrix
        round_converged = (
            round_no == 1
            or high_priority_gaps_reduced
            or requirement_coverage_delta > 0
            or (not requirement_matrix and independent_source_delta > 0)
        )
        if decision.decision == "need_more" and not can_continue:
            decision = decision.model_copy(
                update={
                    "decision": "partial",
                    "reasoning": [
                        *decision.reasoning,
                        "Research soft deadline or remaining-capacity gate reached; finalize from the current evidence ledger.",
                    ],
                }
            )
        if decision.decision == "need_more" and not round_converged:
            decision = decision.model_copy(
                update={
                    "decision": "partial",
                    "reasoning": [
                        *decision.reasoning,
                        "The completed round neither reduced high-priority gaps nor filled a new target requirement.",
                    ],
                }
            )
        round_progress.update(
            {
                "current_high_priority_gap_count": current_high_priority_gap_count,
                "high_priority_gaps_reduced": high_priority_gaps_reduced,
                "round_converged": round_converged,
            }
        )
        result = {
            "synthesis_decision": decision.model_dump(mode="json"),
            "synthesis_traces": [trace.model_dump(mode="json")],
            "unresolved_gaps": [
                item.model_dump(mode="json") for item in decision.missing_information
            ],
            "events": [
                self._event(
                    state,
                    "synthesis_completed",
                    170 + (round_no - 1) * 200,
                    payload={
                        "round_no": round_no,
                        "decision": decision.decision,
                        "source_context_available": trace.source_context_available,
                        "gap_count": len(decision.missing_information),
                        "evidence_delta": evidence_delta,
                        "source_delta": source_delta,
                        "round_progress": round_progress,
                    },
                )
            ],
        }
        if response is not None:
            result["usage_events"] = [
                usage_event(
                    state["run_id"],
                    None,
                    "llm",
                    round_no * 10 + 2,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ).model_dump()
            ]
        emit_progress(
            "synthesizer_completed",
            phase="synthesis",
            agent_role="synthesizer",
            status="completed",
            message=(
                f"第 {round_no} 轮评估完成：{decision.decision}，"
                f"关键缺口 {len(decision.missing_information)} 个"
            ),
            operation_id=operation_id,
            round_no=round_no,
            visibility="quiet",
            metrics={
                "decision": decision.decision,
                "gap_count": len(decision.missing_information),
                "evidence_delta": evidence_delta,
                "source_delta": source_delta,
                "round_progress": round_progress,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "duration_seconds": trace.duration_seconds,
            },
        )
        return result

    @staticmethod
    def _route_synthesis(state: ResearchState) -> str:
        if state.get("synthesis_decision", {}).get("decision") == "need_more":
            return "plan_research"
        return "build_structured_output"

    async def _build_output(self, state: ResearchState) -> dict:
        operation_id = "writer"
        emit_progress(
            "writer_started",
            phase="writing",
            agent_role="writer",
            status="started",
            message="Writer 正在生成最终研究报告",
            operation_id=operation_id,
            visibility="quiet",
        )
        output, citations, response, report_trace = await self.synthesizer.write(state)
        result = {
            "structured_output": output.model_dump(mode="json"),
            "citations": [item.model_dump() for item in citations],
            "writer_trace": report_trace.model_dump(mode="json"),
            "status": output.run_status.value,
            "events": [
                self._event(
                    state,
                    "report_synthesized",
                    700,
                    status=output.run_status.value,
                    payload={"claim_count": len(state.get("claims", []))},
                )
            ],
        }
        if response is not None:
            result["usage_events"] = [
                usage_event(
                    state["run_id"],
                    None,
                    "llm",
                    702,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ).model_dump()
            ]
        emit_progress(
            "writer_completed",
            phase="writing",
            agent_role="writer",
            status="completed",
            message=f"报告正文生成完成：{len(output.sections)} 个章节",
            operation_id=operation_id,
            visibility="quiet",
            metrics={
                "section_count": len(output.sections),
                "citation_count": len(citations),
                "fallback_used": report_trace.fallback_used,
                "input_tokens": report_trace.input_tokens,
                "output_tokens": report_trace.output_tokens,
                "writer_duration_seconds": report_trace.writer_duration_seconds,
                "reviewer_duration_seconds": report_trace.reviewer_duration_seconds,
            },
        )
        return result

    async def _citation_gate(self, state: ResearchState) -> dict:
        """Attach sources and apply the profile-selected citation policy."""
        output = StructuredOutput.model_validate(state["structured_output"])
        sources = [SourceRecord.model_validate(item) for item in state.get("sources", [])]
        evidence = [EvidenceRecord.model_validate(item) for item in state.get("evidence", [])]
        snapshots = [
            SourceSnapshot.model_validate(item)
            for item in state.get("source_snapshots", [])
        ]
        claims = [ClaimRecord.model_validate(item) for item in state.get("claims", [])]
        links = [ClaimEvidenceLink.model_validate(item) for item in state.get("links", [])]
        citations = [CitationRecord.model_validate(item) for item in state.get("citations", [])]
        profile = self.profiles.get(state["request"].get("profile_id", "general_research"))
        citation_policy = profile.runtime_policy.semantic_citation_gate
        strict_citations = citation_policy == "strict"
        citation_gate_enforced = citation_policy in {"repair", "strict"}
        # Third gate for library sources: refetch the authoritative publication state
        # immediately before publication, even when prose citations are advisory.
        from .knowledge.verification import invalid_library_sources, exclude_stale_citations
        cited_links = {c.link_id for c in citations}
        cited_evidence = {link.evidence_id for link in links if link.link_id in cited_links}
        cited_sources = {ev.source_id for ev in evidence if ev.evidence_id in cited_evidence}
        invalid_library = await asyncio.to_thread(
            invalid_library_sources, getattr(getattr(self, "worker", None), "knowledge", None),
            [source for source in sources if source.source_id in cited_sources], snapshots)
        invalid_evidence = {ev.evidence_id for ev in evidence if ev.source_id in invalid_library}
        invalid_links = {link.link_id for link in links if link.evidence_id in invalid_evidence}
        stale_citations = {c.citation_id for c in citations if c.link_id in invalid_links}
        if stale_citations:
            output = exclude_stale_citations(output, stale_citations)
            citations = [c for c in citations if c.citation_id not in stale_citations]
        gate = validate_citations(
            output,
            sources,
            snapshots,
            evidence,
            claims,
            links,
            citations,
            require_all_factual_citations=strict_citations,
        )
        status = output.run_status
        if invalid_library:
            gate = gate.model_copy(update={"passed": False, "violations": [
                *gate.violations, *[f"Library source is archived, changed, or unavailable: {sid}" for sid in sorted(invalid_library)]]})
            if status == RunStatus.COMPLETED_VERIFIED:
                status = RunStatus.COMPLETED_PARTIAL
                output = output.model_copy(update={"run_status": status})
        if (
            citation_gate_enforced
            and not gate.passed
            and status == RunStatus.COMPLETED_VERIFIED
        ):
            status = RunStatus.COMPLETED_PARTIAL
            output = output.model_copy(update={"run_status": status})
        if strict_citations and not gate.passed:
            output = apply_citation_gate(output, gate)
        markdown = render_markdown(
            output,
            citations,
            links,
            evidence,
            sources,
            include_audit_links=profile.runtime_policy.include_audit_links,
        )
        emit_progress(
            "citations_completed",
            phase="citation",
            agent_role="citation",
            status="completed",
            message=f"引用关系已附加：{len(citations)} 条",
            visibility="normal",
            metrics={"citation_count": len(citations)},
        )
        return {
            "structured_output": output.model_dump(mode="json"),
            "gate_result": gate.model_dump(),
            "markdown": markdown,
            "status": status.value,
            "events": [
                self._event(
                    state,
                    "citations_attached",
                    800,
                    status=status.value,
                    payload={"citation_count": len(citations)},
                )
            ],
        }

    async def _finalize_run(self, state: ResearchState) -> dict:
        return {
            "events": [
                self._event(
                    state,
                    "run_terminal",
                    900,
                    status=state["status"],
                    payload={"usage": summarize_usage(state.get("usage_events", []))},
                )
            ]
        }

    async def _collect_skill_evolution(self, state: ResearchState) -> dict:
        """Observe completed state without changing any research decision."""

        runtime_config = self.planner.runtime_config
        config = runtime_config.skill_evolution
        request = ResearchRequest.model_validate(state["request"])
        if runtime_config.effective_skill_evolution_mode(request.profile_id) == "off":
            return {}
        try:
            observer = SkillEvolutionObserver(
                self.worker.skills,
                duplicate_query_threshold=config.duplicate_query_threshold,
                repeated_domain_threshold=config.repeated_domain_threshold,
                max_signals_per_run=config.max_signals_per_run,
            )
            signals, episodes = observer.observe(state)
            store_path = ""
            if self.evolution_store is not None:
                store_path = str(
                    self.evolution_store.write_run(state["run_id"], episodes)
                )
            observation = {
                "mode": "observe",
                "status": "completed",
                "signal_count": len(signals),
                "episode_count": len(episodes),
                "store_path": store_path,
                "candidate_generation_enabled": False,
                "stable_skill_mutation_enabled": False,
            }
            return {
                "evolution_signals": [item.model_dump(mode="json") for item in signals],
                "evolution_episodes": [item.model_dump(mode="json") for item in episodes],
                "skill_evolution_observation": observation,
                "events": [
                    self._event(
                        state,
                        "skill_evolution_observed",
                        910,
                        status="completed",
                        payload=observation,
                    )
                ],
            }
        except Exception as exc:
            # Skill evolution is an isolated observation side path. It must not
            # turn a successful research run into a failed delivery.
            observation = {
                "mode": "observe",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "candidate_generation_enabled": False,
                "stable_skill_mutation_enabled": False,
            }
            return {
                "skill_evolution_observation": observation,
                "events": [
                    self._event(
                        state,
                        "skill_evolution_observation_failed",
                        910,
                        status="failed",
                        payload=observation,
                    )
                ],
            }

    async def _persist_artifacts(self, state: ResearchState) -> dict:
        operation_id = "artifacts"
        emit_progress(
            "artifacts_started",
            phase="artifact",
            agent_role="artifact",
            status="started",
            message="正在保存报告、来源、Trace 和审计文件",
            operation_id=operation_id,
            visibility="normal",
        )
        artifact_dir, output_path = self.workspace.persist_run(state)
        emit_progress(
            "artifacts_completed",
            phase="artifact",
            agent_role="artifact",
            status="completed",
            message=f"报告已保存：{output_path}",
            operation_id=operation_id,
            visibility="normal",
            metrics={"artifact_dir": artifact_dir, "output_path": output_path},
        )
        return {"artifact_dir": artifact_dir, "output_path": output_path}
