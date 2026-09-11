"""Deterministic metrics computed from the typed evidence ledger."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from deepresearch.ids import normalize_text

from .io import load_jsonl, load_unique_jsonl, unique_records, write_json


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _event_latency(events: list[dict[str, Any]]) -> float | None:
    if len(events) < 2:
        return None
    try:
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in events]
        return max(0.0, (max(timestamps) - min(timestamps)).total_seconds())
    except (KeyError, TypeError, ValueError):
        return None


def score_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    details = run_dir / "details"
    evidence_dir = details / "evidence"
    sources = _read_json(evidence_dir / "sources.json", [])
    snapshots = _read_json(evidence_dir / "source-snapshots.json", [])
    evidence = _read_json(evidence_dir / "evidence.json", [])
    claims = _read_json(evidence_dir / "claims.json", [])
    links = _read_json(evidence_dir / "claim-evidence-links.json", [])
    citations = _read_json(evidence_dir / "citations.json", [])
    gate = _read_json(evidence_dir / "citation-assessment.json", {"passed": False})
    usage_events = _read_json(details / "runtime" / "usage.json", [])
    manifest = _read_json(details / "manifest.json", {})
    events = load_jsonl(details / "runtime" / "events.jsonl")

    snapshot_map = {item["source_id"]: item for item in snapshots}
    evidence_map = {item["evidence_id"]: item for item in evidence}
    quote_valid: set[str] = set()
    for item in evidence:
        snapshot = snapshot_map.get(item.get("source_id"))
        if snapshot and normalize_text(item.get("quote", "")) in normalize_text(snapshot.get("content", "")):
            quote_valid.add(item["evidence_id"])

    supported_claims = {
        item["claim_id"]
        for item in links
        if item.get("relation") == "supports"
        and item.get("entailment_status") == "verified"
        and item.get("evidence_id") in quote_valid
    }
    high_claims = {item["claim_id"] for item in claims if item.get("impact") == "high"}
    grounded = set(gate.get("grounded_citation_ids", []))
    unique_usage_rows = unique_records(
        usage_events,
        key_fields=("usage_event_id",),
        record_name="usage",
    )
    usage = {
        "input_tokens": sum(item.get("input_tokens", 0) for item in unique_usage_rows),
        "output_tokens": sum(item.get("output_tokens", 0) for item in unique_usage_rows),
        "searches": sum(item.get("searches", 0) for item in unique_usage_rows),
        "fetches": sum(item.get("fetches", 0) for item in unique_usage_rows),
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    budget = manifest.get("budget", {})
    budget_compliant = all(
        usage[value] <= budget.get(limit, math.inf)
        for value, limit in (
            ("total_tokens", "max_total_tokens"),
            ("searches", "max_total_searches"),
            ("fetches", "max_total_fetches"),
        )
    )
    terminal_count = sum(item.get("event_type") == "run_terminal" for item in events)
    status = manifest.get("status", "unknown")
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "status": status,
        "verified": status == "completed_verified",
        "partial": status == "completed_partial",
        "failed": status.startswith("failed") or status == "cancelled",
        "citation_gate_passed": bool(gate.get("passed")),
        "citation_grounding_precision": len(grounded) / len(citations) if citations else 0.0,
        "quote_match_rate": len(quote_valid) / len(evidence) if evidence else 0.0,
        "unsupported_claim_rate": (
            (len(claims) - len({item["claim_id"] for item in claims} & supported_claims)) / len(claims)
            if claims
            else 0.0
        ),
        "high_impact_claim_coverage": (
            len(high_claims & supported_claims) / len(high_claims) if high_claims else 1.0
        ),
        "provenance_completeness": (
            sum(
                bool(item.get("canonical_uri") and item.get("fetched_at") and item.get("content_hash"))
                for item in sources
            )
            / len(sources)
            if sources
            else 0.0
        ),
        "budget_compliant": budget_compliant,
        "terminal_event_exactly_once": terminal_count == 1,
        "latency_seconds": _event_latency(events),
        **usage,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    latencies = [float(item["latency_seconds"]) for item in rows if item.get("latency_seconds") is not None]
    return {
        "total_runs": total,
        "verified_run_rate": sum(bool(item.get("verified")) for item in rows) / total if total else 0.0,
        "partial_run_rate": sum(bool(item.get("partial")) for item in rows) / total if total else 0.0,
        "failure_rate": sum(bool(item.get("failed")) for item in rows) / total if total else 0.0,
        "citation_grounding_precision": _mean(rows, "citation_grounding_precision"),
        "quote_match_rate": _mean(rows, "quote_match_rate"),
        "unsupported_claim_rate": _mean(rows, "unsupported_claim_rate"),
        "high_impact_claim_coverage": _mean(rows, "high_impact_claim_coverage"),
        "provenance_completeness": _mean(rows, "provenance_completeness"),
        "budget_compliance_rate": _mean(rows, "budget_compliant"),
        "terminal_event_exactly_once_rate": _mean(rows, "terminal_event_exactly_once"),
        "mean_total_tokens": _mean(rows, "total_tokens"),
        "mean_searches": _mean(rows, "searches"),
        "mean_fetches": _mean(rows, "fetches"),
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
    }


def score_evaluation(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    predictions = load_unique_jsonl(
        run_dir / "predictions.jsonl",
        key_fields=("case_id", "repetition"),
        record_name="prediction",
    )
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        artifact_dir = prediction.get("artifact_dir")
        if artifact_dir and Path(artifact_dir).exists():
            score = score_run_artifacts(artifact_dir)
        else:
            score = {
                "run_id": prediction.get("run_id"),
                "status": prediction.get("status", "failed_system"),
                "verified": False,
                "partial": False,
                "failed": True,
                "citation_gate_passed": False,
                "budget_compliant": False,
                "terminal_event_exactly_once": False,
                "latency_seconds": prediction.get("latency_seconds"),
            }
        score.update(
            {
                "evaluation_id": prediction.get("evaluation_id"),
                "case_id": prediction.get("case_id"),
                "repetition": prediction.get("repetition"),
            }
        )
        rows.append(score)

    scores_path = run_dir / "deterministic_scores.jsonl"
    scores_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in rows),
        encoding="utf-8",
    )
    summary = summarize_scores(rows)
    write_json(run_dir / "summary.json", summary)
    return summary
