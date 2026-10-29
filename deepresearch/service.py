"""Dependency assembly and public async service."""

from __future__ import annotations

import os
import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from langgraph.checkpoint.memory import InMemorySaver

from .agents import Planner, ResearchWorker, Synthesizer
from .artifacts import ArtifactWorkspace
from .budget import summarize_usage
from .clarifier import Clarifier
from .graph import ResearchGraph
from .ids import make_id
from .llm import ResearchLLM
from .memory import LayeredMemory, SessionStore
from .mcp import MCPClientManager
from .model_routing import ModelRouter
from .progress import (
    ProgressReporter,
    bind_progress_ids,
    emit_progress,
    progress_context,
)
from .profiles import ProfileRegistry
from .retrieval import SafeWebFetcher, SearchBackend
from .runtime_config import RuntimeConfig, load_runtime_config, resolve_runtime_config_path
from .schemas import (
    ChatResult,
    GateResult,
    MemoryEpisode,
    MemoryPacket,
    ProgressEvent,
    ResearchRequest,
    ResearchMode,
    RunBudget,
    RunResult,
    RunStatus,
)
from .skill_loader import load_skill_registry
from .skill_evolution import (
    CandidateReplayObservation,
    CandidateReplayResult,
    CandidateEvolutionService,
    CandidateGenerationResult,
    CandidateStore,
    EvolutionEpisodeStore,
    VerifierCoEvolutionService,
    VerifierEvolutionStore,
    VerifierSuiteGenerationResult,
    VerifierSuiteRelease,
)
from .state import merge_events, merge_usage
from .worker_checkpoints import WorkerCheckpointStore


class ResearchService:
    def __init__(
        self,
        *,
        llm: ResearchLLM | None = None,
        search_backend: SearchBackend | None = None,
        fetcher: SafeWebFetcher | None = None,
        artifact_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        checkpointer: Any | None = None,
        config_path: str | Path | None = None,
        runtime_config: RuntimeConfig | None = None,
        session_store: SessionStore | None = None,
        candidate_generator_llm: ResearchLLM | None = None,
        candidate_verifier_llm: ResearchLLM | None = None,
        verifier_evolution_llm: ResearchLLM | None = None,
    ):
        if config_path is not None and runtime_config is not None:
            raise ValueError("Pass either config_path or runtime_config, not both")
        self.llm = llm
        self.search_backend = search_backend
        self.fetcher = fetcher
        self.artifact_root = Path(
            artifact_root or os.getenv("DEEPRESEARCH_ARTIFACT_ROOT", "artifacts")
        )
        self.checkpoint_path = Path(
            checkpoint_path
            or os.getenv("DEEPRESEARCH_CHECKPOINT_DB", ".runtime/checkpoints.sqlite")
        )
        self.injected_checkpointer = checkpointer
        self.runtime_config = runtime_config or load_runtime_config(config_path)
        self.runtime_config_dir = resolve_runtime_config_path(config_path).parent
        self.injected_session_store = session_store
        self.injected_candidate_generator_llm = candidate_generator_llm
        self.injected_candidate_verifier_llm = candidate_verifier_llm
        self.injected_verifier_evolution_llm = verifier_evolution_llm
        self._checkpoint_context = None
        self._graph: ResearchGraph | None = None
        self.memory: LayeredMemory | None = None
        self.clarifier: Clarifier | None = None
        self.mcp_manager: MCPClientManager | None = None
        self.model_router: ModelRouter | None = None
        self.candidate_evolution: CandidateEvolutionService | None = None
        self.verifier_coevolution: VerifierCoEvolutionService | None = None

    async def __aenter__(self) -> "ResearchService":
        from .tool_runtime import ToolRuntime
        from .runtime_monitor import RuntimeMonitor
        from .skill_runtime import SkillRuntimeManager
        self.tool_runtime = ToolRuntime(self.runtime_config.governance)
        self.monitor = RuntimeMonitor(self.tool_runtime)
        self.model_router = ModelRouter(
            self.runtime_config,
            injected_llm=self.llm,
        )
        self.memory = LayeredMemory(
            config=self.runtime_config.memory,
            root=Path(__file__).resolve().parent.parent,
            llm=self.model_router.for_stage("memory"),
            session_store=self.injected_session_store,
        )
        await self.memory.open()
        self.clarifier = Clarifier(
            self.model_router.for_stage("clarifier"), self.runtime_config
        )
        profiles = ProfileRegistry(runtime_config=self.runtime_config)
        overrides = {}
        if self.search_backend is not None:
            overrides["web-search"] = self.search_backend.search
        if self.fetcher is not None:
            overrides["safe-fetch"] = self.fetcher.fetch
            overrides["safe-fetch-evaluation"] = self.fetcher.fetch
        skills = load_skill_registry(
            Path(__file__).resolve().parent.parent,
            runtime_overrides=overrides,
            executable_roles=self.runtime_config.tool_permissions.executable_roles,
            profile_skill_fallbacks={
                profile_id: profile.skill_fallbacks
                for profile_id, profile in self.runtime_config.agent_profiles.items()
                if profile.skill_fallbacks
            },
            profile_skill_provider_modes={
                profile_id: profile.skill_provider_modes
                for profile_id, profile in self.runtime_config.agent_profiles.items()
                if profile.skill_provider_modes
            },
            skill_mode=self.runtime_config.graph_engineering.skill_mode,
            layer_settings=self.runtime_config.skill_layers.model_dump(mode="python"),
        )
        # An injected parent search backend fully replaces ``web-search`` (as
        # in tests and embedded deployments), so starting its MCP children is
        # unnecessary and can create external connections that will never run.
        if self.search_backend is None:
            self.mcp_manager = MCPClientManager(
                self.runtime_config.mcp,
                base_dir=self.runtime_config_dir,
            )
            await self.mcp_manager.open()
            try:
                for definition in self.mcp_manager.skill_definitions():
                    skills.register(definition)
            except Exception:
                await self.mcp_manager.close()
                self.mcp_manager = None
                raise

        self.skill_runtime = SkillRuntimeManager(skills)
        checkpointer = self.injected_checkpointer
        if checkpointer is None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_context = AsyncSqliteSaver.from_conn_string(
                str(self.checkpoint_path)
            )
            checkpointer = await self._checkpoint_context.__aenter__()

        evolution_root = (
            Path(self.runtime_config.skill_evolution.store_root)
            if Path(self.runtime_config.skill_evolution.store_root).is_absolute()
            else Path(__file__).resolve().parent.parent
            / self.runtime_config.skill_evolution.store_root
        )
        episode_store = EvolutionEpisodeStore(evolution_root)
        self._graph = ResearchGraph(
            planner=Planner(
                self.model_router.for_stage("planner"), skills, self.runtime_config
            ),
            worker=ResearchWorker(
                self.model_router.for_stage("worker_agent_loop"),
                skills,
                self.runtime_config,
                strategy_llm=self.model_router.for_stage("worker_strategy"),
                reflection_llm=self.model_router.for_stage("worker_reflection"),
                agent_loop_llm=self.model_router.for_stage("worker_agent_loop"),
                extraction_llm=self.model_router.for_stage("evidence_extraction"),
            ),
            synthesizer=Synthesizer(
                self.model_router.for_stage("synthesizer"),
                skills,
                self.runtime_config,
                writer_llm=self.model_router.for_stage("report_writer"),
                reviewer_llm=self.model_router.for_stage("reviewer"),
            ),
            profiles=profiles,
            workspace=ArtifactWorkspace(self.artifact_root),
            evolution_store=episode_store,
            checkpointer=checkpointer,
            worker_checkpoint_store=(
                WorkerCheckpointStore(self.checkpoint_path.with_suffix(".workers.sqlite"))
                if self.injected_checkpointer is None else None
            ),
        )
        candidate_config = self.runtime_config.skill_evolution.candidate_generation
        candidate_store = CandidateStore(evolution_root)
        self.candidate_evolution = CandidateEvolutionService(
            registry=skills,
            episode_store=episode_store,
            candidate_store=candidate_store,
            artifact_root=self.artifact_root,
            generator_llm=(
                self.injected_candidate_generator_llm
                or self.model_router.resolve_tier(candidate_config.generator_tier)
            ),
            verifier_llm=(
                self.injected_candidate_verifier_llm
                or self.model_router.resolve_tier(candidate_config.verifier_tier)
            ),
            config=candidate_config,
        )
        coevolution_config = self.runtime_config.skill_evolution.verifier_coevolution
        self.verifier_coevolution = VerifierCoEvolutionService(
            candidate_service=self.candidate_evolution,
            candidate_store=candidate_store,
            episode_store=episode_store,
            verifier_store=VerifierEvolutionStore(evolution_root),
            generator_llm=(
                self.injected_verifier_evolution_llm
                or self.model_router.resolve_tier(coevolution_config.generator_tier)
            ),
            config=coevolution_config,
        )
        await self.monitor.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        async def cleanup() -> None:
            await self.monitor.close()
            # Stop active external transports before closing the checkpoint;
            # MCP owner tasks may still be finishing an in-flight call.
            try:
                if self.mcp_manager is not None:
                    await self.mcp_manager.close()
            finally:
                try:
                    if self._checkpoint_context is not None:
                        await self._checkpoint_context.__aexit__(exc_type, exc, tb)
                finally:
                    try:
                        if self.memory is not None:
                            await self.memory.close()
                    finally:
                        knowledge = getattr(getattr(self._graph, "worker", None), "knowledge", None)
                        if knowledge is not None and hasattr(knowledge, "close"):
                            await asyncio.to_thread(knowledge.close)

        cleanup_task = asyncio.create_task(cleanup(), name="research-service-close")
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            # Do not interrupt aiosqlite/MCP cleanup with the caller's cancel
            # scope. Await the already-running cleanup task, then propagate.
            await asyncio.shield(cleanup_task)
            raise
        finally:
            self._graph = None
            self._checkpoint_context = None
            self.memory = None
            self.clarifier = None
            self.mcp_manager = None
            self.model_router = None
            self.candidate_evolution = None
            self.verifier_coevolution = None

    async def generate_skill_candidate(
        self,
        episode_id: str,
        *,
        human_review_approved: bool = False,
    ) -> CandidateGenerationResult:
        """Explicitly run phase two for one stored Episode.

        This method is deliberately separate from ``run``: normal research
        never loads, executes, or publishes a Candidate.
        """

        if self.candidate_evolution is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        return await self.candidate_evolution.generate_for_episode(
            episode_id,
            human_review_approved=human_review_approved,
        )

    def evaluate_skill_candidate_replay(
        self, observation: CandidateReplayObservation | dict[str, Any]
    ) -> CandidateReplayResult:
        """Evaluate one externally produced local replay and record verifier gaps."""

        if self.verifier_coevolution is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        return self.verifier_coevolution.evaluate_replay(observation)

    async def generate_verifier_candidate(
        self, gap_id: str
    ) -> VerifierSuiteGenerationResult:
        """Propose an additive, inactive Verifier Suite version for one Gap."""

        if self.verifier_coevolution is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        return await self.verifier_coevolution.generate_verifier_suite(gap_id)

    def activate_verifier_candidate(
        self,
        suite_id: str,
        version: str,
        *,
        human_review_approved: bool = False,
    ) -> VerifierSuiteRelease:
        """Activate a Verifier Suite only through an explicit reviewed action."""

        if self.verifier_coevolution is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        return self.verifier_coevolution.activate_verifier_suite(
            suite_id,
            version,
            human_review_approved=human_review_approved,
        )

    async def revise_skill_candidate_from_verifier_gap(
        self, gap_id: str
    ) -> CandidateGenerationResult:
        """Re-enter the Skill loop after an evolved Verifier Suite is active."""

        if self.verifier_coevolution is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        return await self.verifier_coevolution.revise_candidate_from_gap(gap_id)

    def _resolve_request_mode(self, request: ResearchRequest) -> ResearchRequest:
        mode_name = self.runtime_config.normalize_mode(request.mode.value)
        option_values = self.runtime_config.resolve_research_options(
            request.options, mode_name, request.profile_id
        )
        return request.model_copy(
            update={
                "mode": ResearchMode(mode_name),
                "options": request.options.model_copy(update=option_values),
            }
        )

    def _default_budget(self, mode: ResearchMode, profile_id: str) -> RunBudget:
        selected = self.runtime_config.effective_mode_config(mode.value, profile_id)
        return RunBudget(
            max_total_searches=selected.max_total_searches,
            max_total_fetches=selected.max_total_fetches,
            max_wall_time_seconds=selected.max_wall_time_seconds,
            max_cost_usd=selected.max_cost_usd,
        )

    async def _result_from_state(self, state: dict[str, Any]) -> RunResult:
        request = ResearchRequest.model_validate(state["request"])
        return RunResult(
            run_id=state["run_id"],
            status=RunStatus(state["status"]),
            mode=request.mode,
            output_path=state.get("output_path"),
            markdown=state.get("markdown", ""),
            sources=state.get("sources", []),
            claims=state.get("claims", []),
            gate=GateResult.model_validate(state.get("gate_result", {"passed": False})),
            usage=summarize_usage(state.get("usage_events", [])),
            artifact_dir=state.get("artifact_dir", ""),
            audit_report_path=(
                str(Path(state["artifact_dir"]) / "details" / "audit" / "audit-report.md")
                if state.get("artifact_dir")
                else None
            ),
            audit_log_path=(
                str(Path(state["artifact_dir"]) / "details" / "audit" / "audit-log.json")
                if state.get("artifact_dir")
                else None
            ),
        )

    async def run(
        self,
        request: ResearchRequest | dict,
        *,
        budget: RunBudget | dict | None = None,
        planner_memory: MemoryPacket | dict[str, Any] | None = None,
    ) -> RunResult:
        validated = (
            request
            if isinstance(request, ResearchRequest)
            else ResearchRequest.model_validate(request)
        )
        validated = self._resolve_request_mode(validated)
        if self.model_router is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        token = self.model_router.activate(validated.mode.value)
        try:
            return await self._run_active(
                validated,
                budget=budget,
                planner_memory=planner_memory,
            )
        finally:
            self.model_router.reset(token)

    async def _run_active(
        self,
        request: ResearchRequest | dict,
        *,
        budget: RunBudget | dict | None = None,
        planner_memory: MemoryPacket | dict[str, Any] | None = None,
    ) -> RunResult:
        if self._graph is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        request = (
            request if isinstance(request, ResearchRequest) else ResearchRequest.model_validate(request)
        )
        run_id = request.run_id or f"run-{uuid.uuid4().hex[:16]}"
        request = request.model_copy(update={"run_id": run_id})
        bind_progress_ids(
            session_id=request.session_id,
            turn_id=request.turn_id,
            run_id=run_id,
        )
        config = {"configurable": {"thread_id": run_id}}
        existing = await self._graph.graph.aget_state(config)
        if existing.values:
            raise RunAlreadyExistsError(
                f"Run {run_id!r} already exists; use resume(run_id) to continue or read it"
            )
        budget = (
            self._default_budget(request.mode, request.profile_id)
            if budget is None
            else RunBudget.model_validate(budget)
        )
        initial_state = {
            "run_id": run_id,
            "user_id": request.user_id,
            "project_id": request.project_id,
            "session_id": request.session_id or "",
            "turn_id": request.turn_id or "",
            "request": request.model_dump(mode="json"),
            "p0_ablation": (
                self.runtime_config.graph_engineering.p0_ablation_snapshot()
            ),
            "planner_memory": (
                planner_memory.model_dump(mode="json")
                if isinstance(planner_memory, MemoryPacket)
                else planner_memory or {"role": "planner"}
            ),
            "run_budget": budget.model_dump(),
            "started_at": time.time(),
            "status": RunStatus.QUEUED.value,
            "contributions": [],
            "tasks": [],
            "current_tasks": [],
            "current_wave_tasks": [],
            "wave_no": 0,
            "wave_baseline": {},
            "dag_validation": {},
            "task_runtime": [],
            "wave_history": [],
            "candidate_leads": [],
            "source_leads": [],
            "entity_candidates": [],
            "evidence_bindings": [],
            "evidence_edges": [],
            "failure_memory": [],
            "coverage_changes": [],
            "observation_commits": [],
            "observation_rejections": [],
            "search_schemas": [],
            "schema_revisions": [],
            "sensor_history": [],
            "evolution_signals": [],
            "evolution_episodes": [],
            "skill_evolution_observation": {},
            "frontier_history": [],
            "cell_coverage": {},
            "scope_completeness": {},
            "sensor_decision": {},
            "controller_decision": {},
            "controller_history": [],
            "requirement_status": [],
            "context_projection_traces": [],
            "dependency_context": {},
            "plan_history": [],
            "unresolved_gaps": [],
            "sources": [],
            "source_snapshots": [],
            "evidence": [],
            "claims": [],
            "links": [],
            "citations": [],
            "usage_events": [],
            "events": [],
            "extraction_traces": [],
            "search_traces": [],
            "fetch_traces": [],
            "agent_loop_traces": [],
            "worker_traces": [],
            "planner_traces": [],
            "synthesis_traces": [],
            "planner_trace": {},
            "writer_trace": {},
        }
        state = await self._invoke_graph(
            initial_state,
            config=config,
        )
        await self._remember_run(state)
        result = await self._result_from_state(state)
        emit_progress(
            "run_completed",
            phase="run",
            agent_role="orchestrator",
            status="completed",
            message=f"研究任务完成：{result.status.value}",
            operation_id=f"run:{run_id}",
            run_id=run_id,
            visibility="quiet",
            metrics={
                "status": result.status.value,
                "sources": len(result.sources),
                "claims": len(result.claims),
                "output_path": result.output_path,
                "usage": result.usage,
            },
        )
        return result

    def monitor_summary(self):
        return self.monitor.summary()

    def reload_skills(self, *, approved=False):
        return self.skill_runtime.reload(approved=approved)

    def rollback_skills(self):
        return self.skill_runtime.rollback()

    async def _invoke_graph(self, initial_state, *, config, operation=None):
        from .tool_runtime import bind_runtime, operation_context
        run_id = config["configurable"]["thread_id"]
        workspace = self._graph.workspace
        path = workspace.resolve(run_id, "details/runtime/skill-snapshot.json")
        if initial_state is None and path.exists():
            import json
            skill_snapshot = json.loads(path.read_text())
        else:
            skill_snapshot = self.skill_runtime.snapshot()
            workspace.write_json(run_id, "details/runtime/skill-snapshot.json", skill_snapshot)
        if initial_state is not None:
            initial_state["skill_snapshot_id"] = skill_snapshot["snapshot_id"]
        self.tool_runtime.run_records[run_id] = []
        try:
            with self.skill_runtime.bind(skill_snapshot), bind_runtime(self.tool_runtime), operation_context(
                run_id=run_id, snapshot_id=skill_snapshot["snapshot_id"]
            ):
                return await operation() if operation else await self._graph.graph.ainvoke(initial_state, config=config)
        finally:
            try:
                rows = self.tool_runtime.run_records.get(run_id, [])
                # Each resume segment is immutable: a second execution must not erase the first.
                segment = uuid.uuid4().hex
                workspace.write_jsonl(run_id, f"details/runtime/operations-{segment}.jsonl", rows)
                summary = self.monitor.summary(rows)
                summary["dropped_records"] = self.tool_runtime.dropped.get(run_id, 0)
                workspace.write_json(run_id, f"details/runtime/monitor-{segment}.json", summary)
            finally:
                self.tool_runtime.release_run(run_id)

    async def resume(self, run_id: str) -> RunResult:
        if self._graph is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        ResearchRequest.model_validate(
            {"question": "checkpoint lookup", "run_id": run_id}
        )
        bind_progress_ids(run_id=run_id)
        emit_progress(
            "resume_started",
            phase="run",
            agent_role="orchestrator",
            status="started",
            message=f"正在加载或恢复研究任务：{run_id}",
            operation_id=f"resume:{run_id}",
            visibility="quiet",
        )
        config = {"configurable": {"thread_id": run_id}}
        snapshot = await self._graph.graph.aget_state(config)
        if not snapshot.values:
            raise RunNotFoundError(f"Run {run_id!r} does not exist")
        request = ResearchRequest.model_validate(snapshot.values["request"])
        if self.model_router is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        token = self.model_router.activate(request.mode.value)
        try:
            if snapshot.next:
                state = await self._invoke_graph(None, config=config)
            else:
                state = snapshot.values
        finally:
            self.model_router.reset(token)
        await self._remember_run(state)
        result = await self._result_from_state(state)
        emit_progress(
            "resume_completed",
            phase="run",
            agent_role="orchestrator",
            status="completed",
            message=f"研究任务已加载：{result.status.value}",
            operation_id=f"resume:{run_id}",
            visibility="quiet",
            metrics={"status": result.status.value, "output_path": result.output_path},
        )
        return result

    async def finalize_only(self, run_id: str) -> RunResult:
        if self._graph is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        ResearchRequest.model_validate({"question": "checkpoint lookup", "run_id": run_id})
        config = {"configurable": {"thread_id": run_id}}
        checkpoint = await self._graph.graph.aget_state(config)
        if not checkpoint.values:
            raise RunNotFoundError(f"Run {run_id!r} does not exist")
        return await self._invoke_graph(None, config=config,
                                        operation=lambda: self._finalize_only_active(run_id))

    async def _finalize_only_active(self, run_id: str) -> RunResult:
        """Rebuild Writer/citation/artifact stages from an existing checkpoint."""

        if self._graph is None or self.model_router is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        ResearchRequest.model_validate(
            {"question": "checkpoint lookup", "run_id": run_id}
        )
        config = {"configurable": {"thread_id": run_id}}
        snapshot = await self._graph.graph.aget_state(config)
        if not snapshot.values:
            raise RunNotFoundError(f"Run {run_id!r} does not exist")
        state = dict(snapshot.values)
        for required_field in ("request", "scope", "run_budget"):
            if not state.get(required_field):
                raise ValueError(
                    f"Run {run_id!r} cannot be finalized without {required_field}"
                )
        request = ResearchRequest.model_validate(state["request"])
        # A recovery may happen hours after retrieval. Rebase only the local
        # finalization clock so Writer and Reviewer receive their configured
        # slots without reopening the research budget.
        research_seconds = float(
            state.get("run_budget", {}).get("max_wall_time_seconds") or 0.0
        )
        state["started_at"] = time.time() - research_seconds
        token = self.model_router.activate(request.mode.value)
        checkpoint_updates: dict[str, Any] = {}

        def apply_updates(delta: dict[str, Any]) -> None:
            for key, value in delta.items():
                if key == "events":
                    state[key] = merge_events(state.get(key, []), value)
                    checkpoint_updates[key] = merge_events(
                        checkpoint_updates.get(key, []), value
                    )
                elif key == "usage_events":
                    state[key] = merge_usage(state.get(key, []), value)
                    checkpoint_updates[key] = merge_usage(
                        checkpoint_updates.get(key, []), value
                    )
                else:
                    state[key] = value
                    checkpoint_updates[key] = value

        try:
            apply_updates(await self._graph._build_output(state))
            apply_updates(await self._graph._citation_gate(state))
            apply_updates(await self._graph._finalize_run(state))
            apply_updates(await self._graph._persist_artifacts(state))
            await self._graph.graph.aupdate_state(
                config,
                checkpoint_updates,
                as_node="persist_artifacts",
            )
        finally:
            self.model_router.reset(token)
        return await self._result_from_state(state)

    async def inspect(self, run_id: str) -> RunResult:
        """Read a checkpointed run without continuing or writing memory."""
        if self._graph is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        ResearchRequest.model_validate(
            {"question": "checkpoint lookup", "run_id": run_id}
        )
        config = {"configurable": {"thread_id": run_id}}
        snapshot = await self._graph.graph.aget_state(config)
        if not snapshot.values:
            raise RunNotFoundError(f"Run {run_id!r} does not exist")
        return await self._result_from_state(snapshot.values)

    async def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        request_overrides: dict[str, Any] | None = None,
        budget: RunBudget | dict | None = None,
    ) -> ChatResult:
        if self.model_router is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        mode_value = (request_overrides or {}).get(
            "mode", self.runtime_config.default_mode
        )
        mode_name = self.runtime_config.normalize_mode(str(mode_value))
        token = self.model_router.activate(mode_name)
        try:
            return await self._chat_active(
                message,
                session_id=session_id,
                user_id=user_id,
                request_overrides=request_overrides,
                budget=budget,
            )
        finally:
            self.model_router.reset(token)

    async def _chat_active(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        request_overrides: dict[str, Any] | None = None,
        budget: RunBudget | dict | None = None,
    ) -> ChatResult:
        """Process one turn in a persistent multi-turn research session."""
        if self._graph is None or self.memory is None or self.clarifier is None:
            raise RuntimeError("Use ResearchService as an async context manager")
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        user_id = user_id or self.runtime_config.memory.default_user_id
        session_id = session_id or f"session-{uuid.uuid4().hex[:16]}"
        turn_id = f"turn-{uuid.uuid4().hex[:16]}"
        bind_progress_ids(session_id=session_id, turn_id=turn_id)
        emit_progress(
            "session_started",
            phase="session",
            agent_role="session",
            status="started",
            message=f"已进入会话：{session_id}",
            operation_id=f"session-turn:{turn_id}",
            visibility="verbose",
        )
        session = await self.memory.get_session(user_id, session_id)
        session = await self.memory.append_message(
            session, role="user", content=message, turn_id=turn_id
        )
        clarifier_memory = await self.memory.clarifier_packet(session, message)
        decision = await self.clarifier.clarify(
            message=message,
            session=session,
            memory=clarifier_memory,
        )
        session = session.model_copy(
            update={
                "active_entities": list(
                    dict.fromkeys([*session.active_entities, *decision.active_entities])
                )[-20:],
                "user_corrections": list(
                    dict.fromkeys([*session.user_corrections, *decision.user_corrections])
                )[-30:],
                "preferences": {**session.preferences, **decision.preferences},
            }
        )
        await self.memory.save_session(session)

        if decision.action == "clarify":
            reply = decision.question or "请补充你希望研究的具体范围。"
            session = session.model_copy(update={"pending_clarification": reply})
            session = await self.memory.append_message(
                session, role="assistant", content=reply, turn_id=turn_id
            )
            await self.memory.sync_outbox()
            result = ChatResult(
                user_id=user_id,
                project_id=self.runtime_config.memory.project_id,
                session_id=session_id,
                turn_id=turn_id,
                action="clarify",
                message=reply,
            )
            emit_progress(
                "session_completed",
                phase="session",
                agent_role="session",
                status="completed",
                message="本轮已生成澄清问题",
                operation_id=f"session-turn:{turn_id}",
                visibility="quiet",
                metrics={"action": result.action},
            )
            return result

        if decision.action == "respond_from_context":
            reply = decision.response or await self.clarifier.respond_from_context(
                message=message, session=session, memory=clarifier_memory
            )
            session = session.model_copy(update={"pending_clarification": None})
            await self.memory.append_message(
                session, role="assistant", content=reply, turn_id=turn_id
            )
            await self.memory.sync_outbox()
            result = ChatResult(
                user_id=user_id,
                project_id=self.runtime_config.memory.project_id,
                session_id=session_id,
                turn_id=turn_id,
                action="respond_from_context",
                message=reply,
            )
            emit_progress(
                "session_completed",
                phase="session",
                agent_role="session",
                status="completed",
                message="本轮已根据会话上下文完成回答",
                operation_id=f"session-turn:{turn_id}",
                visibility="quiet",
                metrics={"action": result.action},
            )
            return result

        brief = decision.research_brief
        assert brief is not None
        planner_memory = await self.memory.planner_packet(session, brief.question)
        planner_memory = planner_memory.model_copy(
            update={
                "user_constraints": list(
                    dict.fromkeys(
                        [
                            *planner_memory.user_constraints,
                            *brief.user_constraints,
                            f"Original user request: {message}",
                        ]
                    )
                )
            }
        )
        raw_request = {
            "question": brief.question,
            "user_id": user_id,
            "project_id": self.runtime_config.memory.project_id,
            "session_id": session_id,
            "turn_id": turn_id,
            **(request_overrides or {}),
        }
        # The fixed project namespace cannot be changed through turn overrides.
        raw_request["project_id"] = self.runtime_config.memory.project_id
        raw_request["question"] = brief.question
        request = ResearchRequest.model_validate(raw_request)
        run = await self.run(request, budget=budget, planner_memory=planner_memory)
        reply = run.markdown
        memory_reply = await self.memory.compact_research_message(
            run_id=run.run_id,
            report_path=run.output_path,
            full_content=reply,
        )
        session = await self.memory.get_session(user_id, session_id)
        session = session.model_copy(
            update={
                "pending_clarification": None,
                "current_intent": brief.objective,
                "related_run_ids": list(
                    dict.fromkeys([*session.related_run_ids, run.run_id])
                ),
            }
        )
        await self.memory.append_message(
            session,
            role="assistant",
            content=memory_reply,
            turn_id=turn_id,
            run_id=run.run_id,
            archive_content=reply,
        )
        await self.memory.sync_outbox()
        result = ChatResult(
            user_id=user_id,
            project_id=self.runtime_config.memory.project_id,
            session_id=session_id,
            turn_id=turn_id,
            action="research",
            message=reply,
            run=run,
        )
        emit_progress(
            "session_completed",
            phase="session",
            agent_role="session",
            status="completed",
            message="本轮深度研究及会话保存完成",
            operation_id=f"session-turn:{turn_id}",
            visibility="quiet",
            metrics={"action": result.action, "run_id": run.run_id},
        )
        return result

    async def _stream_with_progress(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        reporter: ProgressReporter,
        heartbeat_seconds: float,
    ) -> AsyncIterator[ProgressEvent | Any]:
        async def execute() -> Any:
            with progress_context(reporter):
                try:
                    return await operation()
                except Exception as exc:
                    reporter.emit(
                        "operation_failed",
                        phase="run",
                        agent_role="orchestrator",
                        status="failed",
                        message=f"执行失败：{type(exc).__name__}: {exc}",
                        visibility="quiet",
                        operation_id="stream-operation",
                    )
                    raise

        task = asyncio.create_task(execute())
        last_visible_event = time.monotonic()
        poll_seconds = (
            min(0.1, max(0.01, heartbeat_seconds))
            if heartbeat_seconds > 0
            else 0.1
        )
        try:
            while not task.done() or not reporter.queue.empty():
                try:
                    async with asyncio.timeout(poll_seconds):
                        event = await reporter.queue.get()
                except TimeoutError:
                    if (
                        not task.done()
                        and heartbeat_seconds > 0
                        and time.monotonic() - last_visible_event >= heartbeat_seconds
                    ):
                        heartbeat = reporter.heartbeat()
                        if heartbeat is not None:
                            # heartbeat() uses the same queue as ordinary events.
                            reporter.queue.get_nowait()
                            last_visible_event = time.monotonic()
                            yield heartbeat
                    continue
                last_visible_event = time.monotonic()
                yield event
            yield await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.done():
                        await asyncio.shield(task)

    async def chat_stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        request_overrides: dict[str, Any] | None = None,
        budget: RunBudget | dict | None = None,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[ProgressEvent | ChatResult]:
        reporter = ProgressReporter(
            timezone=self.runtime_config.timezone,
            session_id=session_id,
        )
        async for item in self._stream_with_progress(
            lambda: self.chat(
                message,
                session_id=session_id,
                user_id=user_id,
                request_overrides=request_overrides,
                budget=budget,
            ),
            reporter=reporter,
            heartbeat_seconds=heartbeat_seconds,
        ):
            yield item

    async def run_stream(
        self,
        request: ResearchRequest | dict,
        *,
        budget: RunBudget | dict | None = None,
        planner_memory: MemoryPacket | dict[str, Any] | None = None,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[ProgressEvent | RunResult]:
        validated = (
            request
            if isinstance(request, ResearchRequest)
            else ResearchRequest.model_validate(request)
        )
        reporter = ProgressReporter(
            timezone=self.runtime_config.timezone,
            session_id=validated.session_id,
        )
        async for item in self._stream_with_progress(
            lambda: self.run(
                validated,
                budget=budget,
                planner_memory=planner_memory,
            ),
            reporter=reporter,
            heartbeat_seconds=heartbeat_seconds,
        ):
            yield item

    async def resume_stream(
        self,
        run_id: str,
        *,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[ProgressEvent | RunResult]:
        reporter = ProgressReporter(timezone=self.runtime_config.timezone)
        reporter.bind_ids(run_id=run_id)
        async for item in self._stream_with_progress(
            lambda: self.resume(run_id),
            reporter=reporter,
            heartbeat_seconds=heartbeat_seconds,
        ):
            yield item

    async def _remember_run(self, state: dict[str, Any]) -> None:
        if self.memory is None:
            return
        request = ResearchRequest.model_validate(state["request"])
        if not request.session_id or state.get("status") in {
            RunStatus.QUEUED.value,
            RunStatus.RUNNING.value,
        }:
            return
        decision = state.get("synthesis_decision", {})
        summary = str(decision.get("answer_summary") or "").strip()
        if not summary:
            summary = "；".join(
                str(item.get("text") or "")
                for item in state.get("claims", [])[:8]
                if item.get("text")
            ) or "本次研究未形成可复用结论。"
        successful_queries = [
            item.get("query")
            for item in state.get("search_traces", [])
            if item.get("query") and item.get("results") and not item.get("error")
        ]
        failed_queries = [
            item.get("query")
            for item in state.get("search_traces", [])
            if item.get("query") and (item.get("error") or not item.get("results"))
        ]
        episode = MemoryEpisode(
            episode_id=make_id("ep", request.user_id, request.session_id, state["run_id"]),
            user_id=request.user_id,
            project_id=request.project_id,
            session_id=request.session_id,
            run_id=state["run_id"],
            research_brief=request.question,
            status=state.get("status", "unknown"),
            summary=summary,
            verified_claim_ids=[
                item.get("claim_id") for item in state.get("claims", []) if item.get("claim_id")
            ],
            open_gaps=[
                str(item.get("missing_information") or item)
                for item in state.get("unresolved_gaps", [])
            ],
            successful_queries=list(dict.fromkeys(successful_queries)),
            failed_queries=list(dict.fromkeys(failed_queries)),
            entities=[],
            user_corrections=[],
            preferences={},
            report_path=state.get("output_path"),
        )
        try:
            session = await self.memory.get_session(request.user_id, request.session_id)
            episode = episode.model_copy(
                update={
                    "entities": session.active_entities,
                    "user_corrections": session.user_corrections,
                    "preferences": session.preferences,
                }
            )
        except Exception:
            # A run remains publishable even if session enrichment is unavailable.
            pass
        await self.memory.save_episode(episode)
        await self.memory.sync_outbox()

    @classmethod
    def in_memory(cls, **kwargs: Any) -> "ResearchService":
        return cls(checkpointer=InMemorySaver(), **kwargs)


class RunAlreadyExistsError(RuntimeError):
    pass


class RunNotFoundError(LookupError):
    pass
