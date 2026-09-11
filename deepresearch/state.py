"""LangGraph state and idempotent reducers."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from .ids import make_id


def _merge_by_keys(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in [*(left or []), *(right or [])]:
        identity = tuple(item.get(key) for key in keys)
        merged[identity] = item
    return list(merged.values())


def _append_only_by_keys(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Keep the first record immutable and version semantic ID collisions."""

    output = [dict(item) for item in (left or [])]
    by_identity = {
        tuple(item.get(key) for key in keys): item
        for item in output
    }
    for raw in right or []:
        item = dict(raw)
        identity = tuple(item.get(key) for key in keys)
        current = by_identity.get(identity)
        if current is None:
            output.append(item)
            by_identity[identity] = item
            continue
        if current == item:
            continue
        primary_key = keys[0]
        original_id = str(item.get(primary_key) or "")
        versioned_id = make_id("revision", original_id, item)
        versioned = {
            **item,
            primary_key: versioned_id,
        }
        versioned_identity = tuple(versioned.get(key) for key in keys)
        if versioned_identity not in by_identity:
            output.append(versioned)
            by_identity[versioned_identity] = versioned
    return output


def merge_contributions(left, right):
    return _merge_by_keys(left, right, ("task_id", "attempt"))


def merge_tasks(left, right):
    return _merge_by_keys(left, right, ("task_id",))


def merge_plans(left, right):
    return _merge_by_keys(left, right, ("round_no",))


def merge_planner_traces(left, right):
    return _merge_by_keys(left, right, ("round_no",))


def merge_synthesis_traces(left, right):
    return _merge_by_keys(left, right, ("round_no",))


def merge_worker_traces(left, right):
    return _merge_by_keys(left, right, ("task_id", "round_no", "phase"))


def merge_sources(left, right):
    return _merge_by_keys(left, right, ("source_id",))


def merge_snapshots(left, right):
    return _merge_by_keys(left, right, ("source_id",))


def merge_evidence(left, right):
    return _merge_by_keys(left, right, ("evidence_id",))


def merge_evidence_append_only(left, right):
    return _append_only_by_keys(left, right, ("evidence_id",))


def merge_claims(left, right):
    merged = _merge_by_keys(left, right, ("claim_id",))
    by_id = {item.get("claim_id"): item for item in merged}
    for item in merged:
        claim_id = item.get("claim_id")
        values = {
            str(value)
            for source in [*(left or []), *(right or [])]
            if source.get("claim_id") == claim_id
            for value in source.get("requirement_ids", [])
            if str(value)
        }
        if values:
            by_id[claim_id] = {**item, "requirement_ids": sorted(values)}
    return list(by_id.values())


def merge_claims_append_only(left, right):
    merged = _append_only_by_keys(left, right, ("claim_id",))
    by_id = {item.get("claim_id"): item for item in merged}
    for item in merged:
        claim_id = item.get("claim_id")
        values = {
            str(value)
            for source in [*(left or []), *(right or [])]
            if source.get("claim_id") == claim_id
            and source.get("text") == item.get("text")
            for value in source.get("requirement_ids", [])
            if str(value)
        }
        if values:
            by_id[claim_id] = {**item, "requirement_ids": sorted(values)}
    return list(by_id.values())


def merge_links(left, right):
    return _merge_by_keys(left, right, ("link_id",))


def merge_links_append_only(left, right):
    return _append_only_by_keys(left, right, ("link_id",))


def merge_citations(left, right):
    return _merge_by_keys(left, right, ("citation_id",))


def merge_usage(left, right):
    return _merge_by_keys(left, right, ("usage_event_id",))


def merge_events(left, right):
    merged = _merge_by_keys(left, right, ("event_id",))
    return sorted(merged, key=lambda item: (item.get("sequence", 0), item.get("event_id", "")))


def merge_extraction_traces(left, right):
    return _merge_by_keys(left, right, ("trace_id",))


def merge_search_traces(left, right):
    return _merge_by_keys(left, right, ("trace_id",))


def merge_fetch_traces(left, right):
    return _merge_by_keys(left, right, ("trace_id",))


def merge_agent_loop_traces(left, right):
    return _merge_by_keys(left, right, ("trace_id",))


def merge_task_runtime(left, right):
    return _merge_by_keys(left, right, ("task_id",))


def merge_wave_history(left, right):
    return _merge_by_keys(left, right, ("wave_id",))


def merge_candidate_leads(left, right):
    return _merge_by_keys(left, right, ("lead_id",))


def merge_source_leads(left, right):
    return _merge_by_keys(left, right, ("lead_id",))


def merge_entity_candidates(left, right):
    return _merge_by_keys(left, right, ("entity_id",))


def merge_evidence_bindings(left, right):
    return _merge_by_keys(left, right, ("binding_id",))


def merge_evidence_bindings_append_only(left, right):
    return _append_only_by_keys(left, right, ("binding_id",))


def merge_evidence_edges(left, right):
    return _merge_by_keys(left, right, ("edge_id",))


def merge_evidence_edges_append_only(left, right):
    return _append_only_by_keys(left, right, ("edge_id",))


def merge_failure_memory(left, right):
    return _merge_by_keys(left, right, ("failure_id",))


def merge_coverage_changes(left, right):
    return _merge_by_keys(left, right, ("change_id",))


def merge_observation_commits(left, right):
    return _merge_by_keys(left, right, ("delta_id",))


def merge_observation_rejections(left, right):
    return _merge_by_keys(left, right, ("rejection_id",))


def merge_search_schemas(left, right):
    return _merge_by_keys(left, right, ("schema_id", "version"))


def merge_schema_revisions(left, right):
    return _merge_by_keys(left, right, ("revision_id",))


def merge_sensor_history(left, right):
    return _merge_by_keys(left, right, ("snapshot_id",))


def merge_frontier_history(left, right):
    return _merge_by_keys(left, right, ("frontier_event_id",))


def merge_controller_history(left, right):
    return _merge_by_keys(left, right, ("wave_id",))


def merge_requirement_status(left, right):
    return _merge_by_keys(left, right, ("requirement_id",))


def merge_context_projection_traces(left, right):
    return _merge_by_keys(left, right, ("trace_id",))


def merge_skill_selection_trace(left, right):
    return _merge_by_keys(left, right, ("task_id", "round_no"))


def merge_evolution_signals(left, right):
    return _merge_by_keys(left, right, ("signal_id",))


def merge_evolution_episodes(left, right):
    return _append_only_by_keys(left, right, ("episode_id",))


class ResearchState(TypedDict, total=False):
    skill_snapshot_id: str
    run_id: str
    user_id: str
    project_id: str
    session_id: str
    turn_id: str
    request: dict[str, Any]
    p0_ablation: dict[str, Any]
    planner_memory: dict[str, Any]
    worker_memory: dict[str, Any]
    scope: dict[str, Any]
    run_budget: dict[str, Any]
    started_at: float
    status: str
    tasks: Annotated[list[dict[str, Any]], merge_tasks]
    current_tasks: list[dict[str, Any]]
    current_wave_tasks: list[dict[str, Any]]
    wave_no: int
    wave_baseline: dict[str, Any]
    wave_budget: dict[str, Any]
    wave_deadline_at: float
    wave_tool_deadline_at: float
    worker_deadline_at: float
    worker_tool_deadline_at: float
    worker_reflection_deadline_at: float
    dag_validation: dict[str, Any]
    task_runtime: Annotated[list[dict[str, Any]], merge_task_runtime]
    wave_history: Annotated[list[dict[str, Any]], merge_wave_history]
    candidate_leads: Annotated[list[dict[str, Any]], merge_candidate_leads]
    source_leads: Annotated[list[dict[str, Any]], merge_source_leads]
    entity_candidates: Annotated[list[dict[str, Any]], merge_entity_candidates]
    evidence_bindings: Annotated[list[dict[str, Any]], merge_evidence_bindings]
    evidence_edges: Annotated[list[dict[str, Any]], merge_evidence_edges]
    failure_memory: Annotated[list[dict[str, Any]], merge_failure_memory]
    coverage_changes: Annotated[list[dict[str, Any]], merge_coverage_changes]
    observation_commits: Annotated[list[dict[str, Any]], merge_observation_commits]
    observation_rejections: Annotated[
        list[dict[str, Any]], merge_observation_rejections
    ]
    search_schemas: Annotated[list[dict[str, Any]], merge_search_schemas]
    schema_revisions: Annotated[list[dict[str, Any]], merge_schema_revisions]
    sensor_history: Annotated[list[dict[str, Any]], merge_sensor_history]
    frontier_history: Annotated[list[dict[str, Any]], merge_frontier_history]
    cell_coverage: dict[str, Any]
    scope_completeness: dict[str, Any]
    sensor_decision: dict[str, Any]
    controller_decision: dict[str, Any]
    controller_history: Annotated[list[dict[str, Any]], merge_controller_history]
    requirement_status: Annotated[list[dict[str, Any]], merge_requirement_status]
    collection_coverage: dict[str, Any]
    context_projection_traces: Annotated[
        list[dict[str, Any]], merge_context_projection_traces
    ]
    skill_selection_trace: Annotated[
        list[dict[str, Any]], merge_skill_selection_trace
    ]
    evolution_signals: Annotated[
        list[dict[str, Any]], merge_evolution_signals
    ]
    evolution_episodes: Annotated[
        list[dict[str, Any]], merge_evolution_episodes
    ]
    skill_evolution_observation: dict[str, Any]
    dependency_context: dict[str, Any]
    current_task: dict[str, Any]
    round_no: int
    current_plan: dict[str, Any]
    plan_history: Annotated[list[dict[str, Any]], merge_plans]
    synthesis_decision: dict[str, Any]
    unresolved_gaps: list[dict[str, Any]]
    contributions: Annotated[list[dict[str, Any]], merge_contributions]
    sources: Annotated[list[dict[str, Any]], merge_sources]
    source_snapshots: Annotated[list[dict[str, Any]], merge_snapshots]
    evidence: Annotated[list[dict[str, Any]], merge_evidence]
    claims: Annotated[list[dict[str, Any]], merge_claims]
    links: Annotated[list[dict[str, Any]], merge_links]
    citations: Annotated[list[dict[str, Any]], merge_citations]
    usage_events: Annotated[list[dict[str, Any]], merge_usage]
    events: Annotated[list[dict[str, Any]], merge_events]
    extraction_traces: Annotated[list[dict[str, Any]], merge_extraction_traces]
    search_traces: Annotated[list[dict[str, Any]], merge_search_traces]
    fetch_traces: Annotated[list[dict[str, Any]], merge_fetch_traces]
    agent_loop_traces: Annotated[list[dict[str, Any]], merge_agent_loop_traces]
    worker_traces: Annotated[list[dict[str, Any]], merge_worker_traces]
    planner_traces: Annotated[list[dict[str, Any]], merge_planner_traces]
    synthesis_traces: Annotated[list[dict[str, Any]], merge_synthesis_traces]
    planner_trace: dict[str, Any]
    writer_trace: dict[str, Any]
    structured_output: dict[str, Any]
    gate_result: dict[str, Any]
    markdown: str
    artifact_dir: str
    output_path: str
    error: str
