"""Role-specific views over complete research state.

The global ledger remains intact.  Projections reduce repeated prompt
serialization and are intentionally queryable rather than destructive.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from .failure_memory import active_failures
from .ids import make_id


ProjectionRole = Literal[
    "planner", "explore_worker", "search_worker", "verify_worker", "writer", "reviewer"
]


def _matches_task(item: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    task_id = str(task.get("task_id") or "")
    dependencies = {str(value) for value in task.get("depends_on", [])}
    target_entities = {str(value) for value in task.get("target_entity_ids", [])}
    target_fields = {str(value) for value in task.get("target_field_keys", [])}
    requirement_ids = {
        str(value)
        for key in ("requirement_ids", "consumes_requirement_ids", "produces_requirement_ids")
        for value in task.get(key, [])
    }
    return bool(
        not task_id
        or str(item.get("task_id") or "") in {task_id, *dependencies}
        or str(item.get("entity_id") or "") in target_entities
        or str(item.get("field_key") or "") in target_fields
        or requirement_ids
        & {str(value) for value in item.get("requirement_ids", [])}
    )


def _relevant(items: list[dict[str, Any]], task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return task-matched items; no semantic top-N cutoff is applied."""

    return [dict(item) for item in items if _matches_task(item, task)]


def _artifact_index(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose all IDs as compact navigation data without serializing full payloads."""

    index: list[dict[str, Any]] = []
    for key, id_key, label_keys in (
        ("source_leads", "lead_id", ("title", "identifier")),
        ("entity_candidates", "entity_id", ("canonical_label", "entity_type")),
        ("sources", "source_id", ("title", "canonical_uri")),
        ("evidence", "evidence_id", ("source_id",)),
        ("claims", "claim_id", ("text",)),
        ("evidence_bindings", "binding_id", ("entity_id", "field_key")),
        ("failure_memory", "failure_id", ("failure_type", "action_signature")),
    ):
        for item in state.get(key, []):
            identity = str(item.get(id_key) or "")
            if not identity:
                continue
            index.append(
                {
                    "artifact_type": key,
                    "artifact_id": identity,
                    "labels": [
                        str(item.get(field) or "")[:240]
                        for field in label_keys
                        if str(item.get(field) or "")
                    ],
                    "task_id": str(item.get("task_id") or ""),
                    "requirement_ids": [str(value) for value in item.get("requirement_ids", [])],
                }
            )
    return index


def _worker_projection(state: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    """Project a relational closure around the assigned task.

    The Worker receives every matching artifact, plus compact IDs for the
    remaining ledger. This preserves autonomy without repeating unrelated page
    bodies and claims in every Worker prompt.
    """

    source_leads = _relevant(state.get("source_leads", []), task)
    entities = _relevant(state.get("entity_candidates", []), task)
    bindings = _relevant(state.get("evidence_bindings", []), task)
    claims = _relevant(state.get("claims", []), task)
    evidence = _relevant(state.get("evidence", []), task)
    source_ids = {
        str(value)
        for item in [*source_leads, *entities]
        for value in [*item.get("origin_source_ids", []), item.get("origin_source_id", "")]
        if str(value)
    }
    entity_ids = {str(item.get("entity_id")) for item in entities if item.get("entity_id")}
    entity_ids.update(str(item.get("entity_id")) for item in bindings if item.get("entity_id"))
    evidence_ids = {str(item.get("evidence_id")) for item in bindings if item.get("evidence_id")}
    evidence_ids.update(str(item.get("evidence_id")) for item in evidence if item.get("evidence_id"))
    claim_ids = {str(item.get("claim_id")) for item in claims if item.get("claim_id")}
    links = [dict(item) for item in state.get("links", [])]
    claim_ids.update(
        str(item.get("claim_id"))
        for item in links
        if str(item.get("evidence_id")) in evidence_ids and item.get("claim_id")
    )
    requirement_ids = {
        str(value)
        for key in ("requirement_ids", "consumes_requirement_ids", "produces_requirement_ids")
        for value in task.get(key, [])
        if str(value)
    }
    for item in state.get("evidence", []):
        if requirement_ids & {str(value) for value in item.get("requirement_ids", [])}:
            evidence_ids.add(str(item.get("evidence_id")))
    for item in state.get("claims", []):
        if requirement_ids & {str(value) for value in item.get("requirement_ids", [])}:
            claim_ids.add(str(item.get("claim_id")))
    evidence = [
        dict(item)
        for item in state.get("evidence", [])
        if str(item.get("evidence_id")) in evidence_ids
    ]
    claims = [
        dict(item)
        for item in state.get("claims", [])
        if str(item.get("claim_id")) in claim_ids
    ]
    links = [
        item
        for item in links
        if str(item.get("claim_id")) in claim_ids
        or str(item.get("evidence_id")) in evidence_ids
    ]
    source_ids.update(str(item.get("source_id")) for item in evidence if item.get("source_id"))
    sources = [
        dict(item)
        for item in state.get("sources", [])
        if str(item.get("source_id")) in source_ids
    ]
    failure_records = active_failures(state.get("failure_memory", []))
    return {
        "source_leads": source_leads,
        "entity_candidates": entities,
        "evidence_bindings": bindings,
        "claims": claims,
        "links": links,
        "evidence": evidence,
        "sources": sources,
        "failure_guidance": _relevant(
            [item.model_dump(mode="json") for item in failure_records], task
        ),
        "artifact_index": _artifact_index(state),
    }


def project_context(
    *,
    role: ProjectionRole,
    state: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact role view while preserving artifact IDs for drill-down."""

    task = dict(task or {})
    common = {
        "run_id": state.get("run_id"),
        "request": state.get("request", {}),
        "scope": state.get("scope", {}),
        "search_schema": (state.get("search_schemas") or [{}])[-1],
        "cell_coverage": state.get("cell_coverage", {}),
        "scope_completeness": state.get("scope_completeness", {}),
    }
    if role == "planner":
        return {
            **common,
            "requirement_status": state.get("requirement_status", []),
            "unresolved_gaps": state.get("unresolved_gaps", []),
            "artifact_counts": artifact_counts(state),
            "active_failures": state.get("failure_memory", []),
            "prior_plan_summaries": state.get("plan_history", []),
        }

    if role in {"explore_worker", "search_worker", "verify_worker"}:
        projected = {
            **common,
            "task_contract": task,
            **_worker_projection(state, task),
        }
        if role == "explore_worker":
            projected.pop("evidence_bindings", None)
        elif role == "search_worker":
            projected["missing_field_keys"] = list(task.get("target_field_keys", []))
        elif role == "verify_worker":
            projected["conflicting_bindings"] = [
                item
                for item in state.get("evidence_bindings", [])
                if item.get("status") in {"conflicting", "uncertain"}
                and _matches_task(item, task)
            ]
        return projected

    if role == "writer":
        return {
            **common,
            "materialized_entities": [
                item
                for item in state.get("entity_candidates", [])
                if item.get("materialization_status") == "materialized"
            ],
            "evidence_bindings": state.get("evidence_bindings", []),
            "claims": state.get("claims", []),
            "evidence": state.get("evidence", []),
            "sources": state.get("sources", []),
            "unresolved_gaps": state.get("unresolved_gaps", []),
        }

    return {
        **common,
        "structured_output": state.get("structured_output", {}),
        "evidence_bindings": state.get("evidence_bindings", []),
        "claims": state.get("claims", []),
        "links": state.get("links", []),
        "citations": state.get("citations", []),
        "unresolved_gaps": state.get("unresolved_gaps", []),
    }


def artifact_counts(state: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: len(state.get(key, []))
        for key in (
            "source_leads",
            "entity_candidates",
            "sources",
            "evidence",
            "claims",
            "evidence_bindings",
            "failure_memory",
        )
    }


def projection_trace(
    *,
    run_id: str,
    role: ProjectionRole,
    task_id: str,
    projection: Mapping[str, Any],
    round_no: int,
) -> dict[str, Any]:
    counts = {
        key: len(value)
        for key, value in projection.items()
        if isinstance(value, list)
    }
    return {
        "trace_id": make_id("projection", run_id, role, task_id, round_no),
        "task_id": task_id,
        "round_no": round_no,
        "role": role,
        "list_counts": counts,
        "full_state_mutated": False,
    }
