#!/usr/bin/env python3
"""Run one frozen-text extraction call through the current entity pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.agents import _build_extraction_prompt, _with_skill_instructions
from deepresearch.evidence import filter_extracted_claims, records_from_document
from deepresearch.model_routing import ModelRouter
from deepresearch.observation_middleware import (
    delta_from_worker_result,
    observations_from_extraction,
)
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import (
    EvidenceExtractionResult,
    FetchedDocument,
    ResearchTask,
    SearchSchemaDefinition,
)
from deepresearch.search_state import (
    commit_observation_delta,
    prune_invalid_observation_items,
)
from deepresearch.skill_loader import load_skill_registry


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: str | Path) -> Any:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _entity_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in claims
        if item.get("entity_id")
        and item.get("entity_label")
        and item.get("entity_type")
    ]


def analyze_extraction_pipeline(
    *,
    run_id: str,
    task: ResearchTask,
    document: FetchedDocument,
    parsed: EvidenceExtractionResult,
    schema: SearchSchemaDefinition,
) -> dict[str, Any]:
    """Pass one parsed model response through the production middleware."""
    parsed_claims = [item.model_dump(mode="json") for item in parsed.claims]
    accepted, rejected = filter_extracted_claims(document, parsed_claims)
    source, snapshot, evidence, claims, links = records_from_document(
        document,
        "frozen_task75_snapshot",
        accepted,
        requirement_ids=task.requirement_ids,
    )
    typed = observations_from_extraction(
        run_id=run_id,
        task=task,
        source=source,
        parsed=parsed,
        accepted_candidates=accepted,
        evidence=evidence,
        schema=schema,
    )
    worker_result = {
        "sources": [source.model_dump(mode="json")],
        "source_snapshots": [snapshot.model_dump(mode="json")],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "claims": [item.model_dump(mode="json") for item in claims],
        "links": [item.model_dump(mode="json") for item in links],
        **typed,
    }
    delta = delta_from_worker_result(
        run_id=run_id,
        task=task,
        result=worker_result,
    )
    initial_state = {
        "run_id": run_id,
        "search_schemas": [schema.model_dump(mode="json")],
    }
    pruned_delta, pruning = prune_invalid_observation_items(initial_state, delta)
    committed = commit_observation_delta(initial_state, pruned_delta)
    state = {
        "run_id": run_id,
        "search_schemas": [schema.model_dump(mode="json")],
        **committed["state_patch"],
    }
    raw_entity_claims = _entity_claims(parsed_claims)
    filtered_entity_claims = _entity_claims(accepted)
    checks = {
        "extractor_emitted_claim": bool(parsed_claims),
        "extractor_emitted_entity_metadata": bool(raw_entity_claims),
        "quotes_survived_filter": bool(accepted),
        "entity_metadata_survived_filter": len(filtered_entity_claims)
        == len(raw_entity_claims),
        "typed_entity_candidate_created": bool(typed["entity_candidates"]),
        "entity_committed_to_state": bool(state.get("entity_candidates")),
        "scope_advanced": state.get("scope_completeness", {}).get("status")
        not in {None, "", "unknown"},
    }
    failed_checks = [key for key, value in checks.items() if not value]
    if raw_entity_claims and not filtered_entity_claims:
        breakpoint = "filter_extracted_claims_drops_entity_metadata"
    elif not raw_entity_claims:
        breakpoint = "extractor_did_not_emit_entity_metadata"
    elif not typed["entity_candidates"]:
        breakpoint = "observations_from_extraction_did_not_create_entity"
    elif not state.get("entity_candidates"):
        breakpoint = "observation_delta_did_not_commit_entity"
    else:
        breakpoint = "none"
    return {
        "status": "pass" if not failed_checks else "fail",
        "breakpoint": breakpoint,
        "checks": checks,
        "failed_checks": failed_checks,
        "observed": {
            "raw_claim_count": len(parsed_claims),
            "raw_entity_claim_count": len(raw_entity_claims),
            "accepted_claim_count": len(accepted),
            "filtered_entity_claim_count": len(filtered_entity_claims),
            "typed_entity_candidate_count": len(typed["entity_candidates"]),
            "committed_entity_count": len(state.get("entity_candidates", [])),
            "scope_status": state.get("scope_completeness", {}).get("status"),
            "rejected_or_warned_claims": rejected,
            "observation_pruning": pruning,
            "post_prune_evidence_count": len(pruned_delta.evidence_records),
            "post_prune_entity_count": len(pruned_delta.entity_candidates),
        },
        "model_output": parsed.model_dump(mode="json"),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_json(args.fixture)
    plan = _load_json(fixture["artifact_paths"]["research_plan"])
    sources = _load_json(fixture["artifact_paths"]["sources"])
    snapshots = _load_json(fixture["artifact_paths"]["source_snapshots"])
    task = ResearchTask.model_validate(
        next(
            item
            for item in plan["tasks"]
            if item.get("task_id") == fixture["task_id"]
        )
    )
    if fixture.get("task_override"):
        task = task.model_copy(update=dict(fixture["task_override"]))
    schema = SearchSchemaDefinition.model_validate(
        next(
            item["search_schema"]
            for item in plan["plans"]
            if item.get("search_schema")
        )
    )
    source = next(
        item for item in sources if item.get("source_id") == fixture["source_id"]
    )
    snapshot = next(
        item
        for item in snapshots
        if item.get("source_id") == fixture["source_id"]
    )
    marker = str(fixture["excerpt_marker"])
    content = str(snapshot["content"])
    positions: list[int] = []
    search_start = 0
    while True:
        match = content.casefold().find(marker.casefold(), search_start)
        if match < 0:
            break
        positions.append(match)
        search_start = match + 1
    if not positions:
        raise ValueError(f"excerpt marker not found: {marker}")
    occurrence_index = int(fixture.get("excerpt_occurrence_index", 0))
    try:
        position = positions[occurrence_index]
    except IndexError as exc:
        raise ValueError(
            f"excerpt occurrence {occurrence_index} is unavailable; found {positions}"
        ) from exc
    start = max(0, position - int(fixture["context_before_chars"]))
    end = min(
        len(content),
        position + len(marker) + int(fixture["context_after_chars"]),
    )
    excerpt = content[start:end]
    document = FetchedDocument(
        requested_url=source["canonical_uri"],
        final_url=source["canonical_uri"],
        title=source["title"],
        content=excerpt,
        content_type=source.get("mime_type") or "text/html",
        http_status=source.get("http_status") or 200,
        fetched_at=source["fetched_at"],
    )
    system, user = _build_extraction_prompt(
        task,
        excerpt,
        source["canonical_uri"],
        requirement_details=[],
        search_schema=schema.model_dump(mode="json"),
    )
    registry = load_skill_registry(ROOT)
    system = _with_skill_instructions(
        system,
        registry,
        role="worker",
        profile_id="drb_evaluation",
        phase="extraction",
    )
    max_tokens = int(args.max_output_tokens)
    token_cap = int(args.token_cap)
    conservative_reservation = len(system.encode()) + len(user.encode()) + max_tokens
    if conservative_reservation > token_cap:
        raise RuntimeError(
            f"token preflight failed: reservation={conservative_reservation}, cap={token_cap}"
        )
    runtime = load_runtime_config(_resolve(args.config))
    llm = ModelRouter(runtime).for_stage("evidence_extraction")
    value, response = await asyncio.wait_for(
        llm.complete_json(system, user, max_tokens=max_tokens),
        timeout=float(args.timeout_seconds),
    )
    actual_tokens = response.input_tokens + response.output_tokens
    if actual_tokens > token_cap:
        raise RuntimeError(
            f"token cap exceeded: actual={actual_tokens}, cap={token_cap}"
        )
    parsed = EvidenceExtractionResult.model_validate(value)
    analysis = analyze_extraction_pipeline(
        run_id=fixture["run_id"],
        task=task,
        document=document,
        parsed=parsed,
        schema=schema,
    )
    return {
        "schema_version": "live-task75-extraction-breakpoint.v1",
        "case_id": fixture["case_id"],
        "scope": "frozen_snapshot_one_extraction_call_no_search_or_fetch",
        "source_id": fixture["source_id"],
        "task_id": fixture["task_id"],
        "excerpt": {
            "marker": marker,
            "occurrence_index": occurrence_index,
            "all_marker_positions": positions,
            "start": start,
            "end": end,
            "chars": len(excerpt),
        },
        "model_id": llm.model_id,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "token_usage": actual_tokens,
        "token_cap": token_cap,
        **analysis,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="config/runtime.yaml")
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--token-cap", type=int, default=50000)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    args = parser.parse_args()
    result = asyncio.run(run(args))
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
