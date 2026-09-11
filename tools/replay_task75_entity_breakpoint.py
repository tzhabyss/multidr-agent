#!/usr/bin/env python3
"""Replay task75's entity handoff from archived Worker output, without research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.ids import make_id, normalize_text
from deepresearch.observation_middleware import delta_from_worker_result
from deepresearch.plan_policy import ready_tasks, task_runtime_seed
from deepresearch.schemas import CandidateLead, EntityCandidate, ObservationDelta, ResearchTask
from deepresearch.search_state import commit_observation_delta


def _resolve(path: str | Path, *, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_json(path: str | Path, *, root: Path = ROOT) -> Any:
    return json.loads(_resolve(path, root=root).read_text(encoding="utf-8"))


def _normalize_reflection_leads(
    *, run_id: str, task: ResearchTask, raw_leads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the same identity and bounded-enum normalization as Worker output."""
    allowed_roles = {
        "source_document",
        "target_entity",
        "entity_container",
        "field_evidence",
        "navigation_only",
        "unknown",
    }
    allowed_types = {"paper", "report", "table", "dataset", "case", "other"}
    allowed_strategies = {
        "bibliography",
        "table_rows",
        "appendix",
        "named_items",
        "dataset_records",
        "follow_links",
        "none",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_leads[:50]:
        title = normalize_text(str(raw.get("title") or ""))
        identifier = normalize_text(str(raw.get("identifier") or ""))
        identity = (title.casefold(), identifier.casefold())
        if not title or identity in seen:
            continue
        seen.add(identity)
        lead_type = str(raw.get("lead_type") or "other")
        artifact_role = str(raw.get("artifact_role") or "unknown")
        extraction_strategy = str(raw.get("extraction_strategy") or "none")
        lead = CandidateLead(
            lead_id=make_id("lead", run_id, task.task_id, title, identifier),
            task_id=task.task_id,
            title=title,
            identifier=identifier,
            lead_type=lead_type if lead_type in allowed_types else "other",
            artifact_role=(
                artifact_role if artifact_role in allowed_roles else "unknown"
            ),
            target_entity_type=normalize_text(
                str(raw.get("target_entity_type") or "")
            ),
            anchor_quote=normalize_text(str(raw.get("anchor_quote") or "")),
            extraction_strategy=(
                extraction_strategy
                if extraction_strategy in allowed_strategies
                else "none"
            ),
            origin_source_ids=[
                str(item) for item in raw.get("origin_source_ids", []) if str(item)
            ],
            requirement_ids=[
                str(item) for item in raw.get("requirement_ids", []) if str(item)
            ],
            missing_fields=[
                normalize_text(str(item))
                for item in raw.get("missing_fields", [])
                if normalize_text(str(item))
            ],
            notes="Archived reflection lead normalized at the Worker breakpoint.",
        )
        normalized.append(lead.model_dump(mode="json"))
    return normalized


def replay_breakpoint(
    fixture: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    traces = _load_json(fixture["artifact_paths"]["worker_traces"], root=root)
    plan = _load_json(fixture["artifact_paths"]["research_plan"], root=root)
    historical_runtime = _load_json(
        fixture["artifact_paths"]["task_runtime"], root=root
    )
    sources = _load_json(fixture["artifact_paths"]["sources"], root=root)
    snapshots = _load_json(fixture["artifact_paths"]["source_snapshots"], root=root)
    evidence = _load_json(fixture["artifact_paths"]["evidence"], root=root)

    tasks = {
        str(item["task_id"]): ResearchTask.model_validate(item)
        for item in plan["tasks"]
    }
    source_task = tasks[fixture["source_task_id"]]
    downstream_task = tasks[fixture["downstream_task_id"]]
    reflection = next(
        row
        for row in traces
        if row.get("phase") == "reflection"
        and row.get("task_id") == source_task.task_id
    )
    run_id = str(reflection["run_id"])
    normalized_leads = _normalize_reflection_leads(
        run_id=run_id,
        task=source_task,
        raw_leads=reflection["raw_output"].get("candidate_leads", []),
    )

    # Breakpoint A: replay only the Worker-output adapter and atomic commit.
    lead_delta = delta_from_worker_result(
        run_id=run_id,
        task=source_task,
        result={"candidate_leads": normalized_leads},
    )
    lead_commit = commit_observation_delta({"run_id": run_id}, lead_delta)
    lead_state = {"run_id": run_id, **lead_commit["state_patch"]}

    normalized_lead_ids = {str(item["lead_id"]) for item in normalized_leads}
    target_lead_ids = {
        str(item["lead_id"])
        for item in normalized_leads
        if item.get("artifact_role") == "target_entity"
    }
    downstream_targets = {str(item) for item in downstream_task.target_entity_ids}
    archived_downstream = next(
        row
        for row in historical_runtime
        if row.get("task_id") == downstream_task.task_id
    )

    # Breakpoint B: ask the current scheduler whether the same downstream task
    # would launch. This is scheduling only; no Worker, search, or fetch runs.
    current_ready, current_blocked = ready_tasks(
        [downstream_task], task_runtime_seed([downstream_task])
    )

    # Breakpoint C: inject one evidence-anchored verified entity into the current
    # atomic commit path. The source/evidence are archived task75 records; only
    # the typed entity observation is simulated.
    anchor_evidence_id = str(fixture["verified_entity"]["anchor_evidence_id"])
    anchor_evidence = next(
        item for item in evidence if item.get("evidence_id") == anchor_evidence_id
    )
    anchor_source_id = str(anchor_evidence["source_id"])
    anchor_source = next(
        item for item in sources if item.get("source_id") == anchor_source_id
    )
    anchor_snapshot = next(
        item for item in snapshots if item.get("source_id") == anchor_source_id
    )
    verified_entity = EntityCandidate(
        entity_id=str(fixture["verified_entity"]["entity_id"]),
        entity_type=str(fixture["verified_entity"]["entity_type"]),
        canonical_label=str(fixture["verified_entity"]["canonical_label"]),
        aliases=list(fixture["verified_entity"].get("aliases", [])),
        origin_source_ids=[anchor_source_id],
        anchor_evidence_ids=[anchor_evidence_id],
        scope_status="in_scope",
        materialization_status="materialized",
        rationale="Simulated verified observation anchored to archived task75 evidence.",
    )
    verification_state = {
        **lead_state,
        "sources": [anchor_source],
        "source_snapshots": [anchor_snapshot],
        "evidence": [anchor_evidence],
    }
    entity_delta = ObservationDelta(
        delta_id=make_id(
            "delta-breakpoint", run_id, downstream_task.task_id, verified_entity.entity_id
        ),
        task_id=downstream_task.task_id,
        entity_candidates=[verified_entity],
    )
    entity_commit = commit_observation_delta(verification_state, entity_delta)
    final_state = {**verification_state, **entity_commit["state_patch"]}
    idempotent_commit = commit_observation_delta(final_state, entity_delta)

    target_source_leads = [
        item
        for item in lead_state.get("source_leads", [])
        if item.get("artifact_role") == "target_entity"
    ]
    checks = {
        "actual_target_entity_leads_present": len(target_lead_ids)
        >= int(fixture["thresholds"]["min_target_entity_leads"]),
        "lead_ids_preserved_by_delta": normalized_lead_ids
        == {str(item.lead_id) for item in lead_delta.source_leads},
        "unverified_leads_not_prematurely_materialized": (
            not lead_delta.entity_candidates
            and not lead_state.get("entity_candidates")
            and len(target_source_leads) == len(target_lead_ids)
        ),
        "downstream_plan_targets_actual_leads": target_lead_ids
        <= downstream_targets,
        "historical_downstream_task_was_blocked": (
            archived_downstream.get("status") == "blocked"
            and archived_downstream.get("stop_reason")
            == "missing_required_artifact_roles"
        ),
        "current_scheduler_releases_same_task": (
            [item.task_id for item in current_ready] == [downstream_task.task_id]
            and not current_blocked
        ),
        "verified_entity_commits": len(final_state.get("entity_candidates", []))
        == 1,
        "scope_advances_after_entity_commit": (
            final_state.get("scope_completeness", {}).get("status") == "open"
            and final_state.get("scope_completeness", {}).get(
                "discovered_entity_count"
            )
            == 1
        ),
        "delta_commit_is_idempotent": (
            idempotent_commit["already_committed"] is True
            and not idempotent_commit["state_patch"]
        ),
    }
    failed_checks = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "task75-entity-breakpoint-replay.v1",
        "case_id": fixture["case_id"],
        "status": "pass" if not failed_checks else "fail",
        "scope": "checkpoint_only_no_search_fetch_or_llm",
        "historical_breakpoint": {
            "task_id": downstream_task.task_id,
            "status": archived_downstream.get("status"),
            "stop_reason": archived_downstream.get("stop_reason"),
            "missing_artifact_roles": archived_downstream.get(
                "missing_artifact_roles", []
            ),
        },
        "observed": {
            "raw_reflection_lead_count": len(
                reflection["raw_output"].get("candidate_leads", [])
            ),
            "normalized_lead_count": len(normalized_leads),
            "target_entity_lead_count": len(target_lead_ids),
            "downstream_target_id_count": len(downstream_targets),
            "matched_downstream_target_count": len(
                normalized_lead_ids & downstream_targets
            ),
            "pre_verification_entity_count": len(
                lead_state.get("entity_candidates", [])
            ),
            "current_ready_task_ids": [item.task_id for item in current_ready],
            "current_blocked_updates": current_blocked,
            "post_verification_entity_count": len(
                final_state.get("entity_candidates", [])
            ),
            "post_verification_scope_status": final_state.get(
                "scope_completeness", {}
            ).get("status"),
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "interpretation": (
            "The archived artifact-role scheduling gate is reproducible, but the "
            "current scheduler releases the same downstream task and the current "
            "atomic commit path materializes an evidence-anchored entity. Remaining "
            "uncertainty is Worker extraction/access behavior, which this checkpoint "
            "simulation intentionally does not execute."
        ),
        "token_usage": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fixture = _load_json(args.fixture)
    result = replay_breakpoint(fixture)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
