"""Recover valid observations pruned by legacy all-or-nothing commits."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepresearch.budget import summarize_usage
from deepresearch.ids import make_id, normalize_text, stable_hash
from deepresearch.schemas import (
    ClaimEvidenceLink,
    ClaimRecord,
    EvidenceRecord,
    ResearchRequest,
)
from deepresearch.service import ResearchService
from deepresearch.state import merge_events, merge_usage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--output-run-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    return parser


def _merge_by_id(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    merged = {
        str(item[key]): dict(item)
        for item in [*existing, *incoming]
        if item.get(key)
    }
    return list(merged.values())


def _recover_records(state: dict[str, Any]) -> dict[str, Any]:
    contribution_claims = {
        str(item.get("task_id")): set(item.get("candidate_claim_ids", []))
        for item in state.get("contributions", [])
    }
    snapshots = {
        str(item.get("source_id")): item
        for item in state.get("source_snapshots", [])
        if item.get("source_id")
    }
    sources_by_uri: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in state.get("sources", []):
        if source.get("canonical_uri") and source.get("source_id") in snapshots:
            sources_by_uri[str(source["canonical_uri"])].append(source)

    evidence: dict[str, dict[str, Any]] = {}
    claims: dict[str, dict[str, Any]] = {}
    links: dict[str, dict[str, Any]] = {}
    valid_submission_pairs: set[tuple[str, str]] = set()
    for trace in state.get("extraction_traces", []):
        task_id = str(trace.get("task_id") or "")
        allowed_claim_ids = contribution_claims.get(task_id, set())
        raw_output = trace.get("raw_output") or {}
        candidates = raw_output.get("claims", []) if isinstance(raw_output, dict) else []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            text = normalize_text(str(candidate.get("text") or ""))
            quote = normalize_text(str(candidate.get("quote") or ""))
            if not text or not quote:
                continue
            claim_id = make_id("clm", text.lower())
            if claim_id not in allowed_claim_ids:
                continue
            matching_sources = []
            for source in sources_by_uri.get(str(trace.get("source_uri") or ""), []):
                snapshot = snapshots[str(source["source_id"])]
                if quote in normalize_text(str(snapshot.get("content") or "")):
                    matching_sources.append(source)
            if not matching_sources:
                continue
            valid_submission_pairs.add((task_id, claim_id))
            requirement_ids = list(
                dict.fromkeys(str(item) for item in candidate.get("requirement_ids", []) if item)
            )
            claims[claim_id] = ClaimRecord(
                claim_id=claim_id,
                text=text,
                impact=str(candidate.get("impact") or "medium"),
                claim_type=str(candidate.get("claim_type") or "factual"),
                status="supported",
                requirement_ids=requirement_ids,
            ).model_dump(mode="json")
            for source in matching_sources:
                source_id = str(source["source_id"])
                evidence_id = make_id(
                    "ev-recovered", source_id, stable_hash(quote), claim_id
                )
                link_id = make_id("lnk-recovered", claim_id, evidence_id, "supports")
                evidence[evidence_id] = EvidenceRecord(
                    evidence_id=evidence_id,
                    source_id=source_id,
                    quote=quote,
                    quote_hash=stable_hash(quote),
                    location={
                        "kind": "recovered_extraction_trace",
                        "locator": str(trace.get("trace_id") or ""),
                    },
                    requirement_ids=requirement_ids,
                ).model_dump(mode="json")
                links[link_id] = ClaimEvidenceLink(
                    link_id=link_id,
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    relation="supports",
                    entailment_status="verified",
                    verification_method="deterministic",
                ).model_dump(mode="json")

    merged_claims = _merge_by_id(state.get("claims", []), list(claims.values()), "claim_id")
    merged_evidence = _merge_by_id(
        state.get("evidence", []), list(evidence.values()), "evidence_id"
    )
    merged_links = _merge_by_id(state.get("links", []), list(links.values()), "link_id")
    return {
        "claims": merged_claims,
        "evidence": merged_evidence,
        "links": merged_links,
        "recovery": {
            "valid_submission_count": len(valid_submission_pairs),
            "unique_claim_count": len(claims),
            "final_claim_count": len(merged_claims),
            "recovered_evidence_count": len(evidence),
            "recovered_link_count": len(links),
        },
    }


async def _run(args: argparse.Namespace) -> None:
    source_dir = args.source_run_dir.resolve()
    output_dir = args.output_run_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output run directory already exists: {output_dir}")
    shutil.copytree(source_dir, output_dir)
    checkpoint_path = output_dir / "checkpoints.sqlite"
    samples_root = output_dir / "samples"

    async with ResearchService(
        artifact_root=samples_root,
        checkpoint_path=checkpoint_path,
    ) as service:
        if service._graph is None or service.model_router is None:
            raise RuntimeError("Research service did not initialize")
        config = {"configurable": {"thread_id": args.run_id}}
        snapshot = await service._graph.graph.aget_state(config)
        if not snapshot.values:
            raise ValueError(f"Checkpoint run not found: {args.run_id}")
        state = dict(snapshot.values)
        recovered = _recover_records(state)
        recovery = recovered.pop("recovery")
        state.update(recovered)
        state["observation_rejections"] = [
            *state.get("observation_rejections", []),
            {
                "rejection_id": make_id("observation-recovery", args.run_id, recovery),
                "delta_id": "legacy-recovery",
                "task_id": "all-rejected-workers",
                "reason": "legacy_all_or_nothing_commit_recovered",
                "action": "valid_items_restored_invalid_items_discarded",
                **recovery,
            },
        ]
        request = ResearchRequest.model_validate(state["request"])
        token = service.model_router.activate(request.mode.value)
        checkpoint_updates: dict[str, Any] = dict(recovered)
        checkpoint_updates["observation_rejections"] = state["observation_rejections"]

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
            apply_updates(await service._graph._synthesize_research(state))
            research_seconds = float(
                state.get("run_budget", {}).get("max_wall_time_seconds") or 0.0
            )
            state["started_at"] = time.time() - research_seconds
            apply_updates(await service._graph._build_output(state))
            apply_updates(await service._graph._citation_gate(state))
            apply_updates(await service._graph._finalize_run(state))
            apply_updates(await service._graph._persist_artifacts(state))
            await service._graph.graph.aupdate_state(
                config, checkpoint_updates, as_node="persist_artifacts"
            )
        finally:
            service.model_router.reset(token)
        result = await service._result_from_state(state)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation_id = output_dir.name
    manifest["evaluation_id"] = evaluation_id
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_path = output_dir / "predictions.jsonl"
    old_prediction = json.loads(prediction_path.read_text(encoding="utf-8").splitlines()[0])
    old_prediction.update(
        {
            "evaluation_id": evaluation_id,
            "status": result.status.value,
            "citation_gate_passed": result.gate.passed,
            "artifact_dir": result.artifact_dir,
            "output_path": result.output_path,
            "usage": summarize_usage(state.get("usage_events", [])),
            "judge_score": None,
            "judge_category_scores": None,
            "judge_error": None,
            "recovery": recovery,
        }
    )
    prediction_path.write_text(
        json.dumps(old_prediction, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name in (
        "judge_results.jsonl",
        "judge_summary.json",
        "deterministic_scores.jsonl",
        "summary.json",
        "evaluation_report.md",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    print(json.dumps(recovery, ensure_ascii=False, indent=2))
    print(result.output_path)


if __name__ == "__main__":
    asyncio.run(_run(_parser().parse_args()))
