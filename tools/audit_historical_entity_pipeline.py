#!/usr/bin/env python3
"""Audit an archived Deep Research run for entity-pipeline failure signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path, *, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _list_count(value: Any, *container_keys: str) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in container_keys:
            items = value.get(key)
            if isinstance(items, list):
                return len(items)
    return 0


def audit_entity_pipeline(
    fixture: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    """Return deterministic pass/fail evidence for one archived run."""
    artifact_paths = fixture["artifact_paths"]
    sources = _load_json(_resolve(artifact_paths["sources"], root=root))
    evidence = _load_json(_resolve(artifact_paths["evidence"], root=root))
    claims = _load_json(_resolve(artifact_paths["claims"], root=root))
    entities = _load_json(_resolve(artifact_paths["entities"], root=root))
    scope = _load_json(_resolve(artifact_paths["scope_completeness"], root=root))
    waves = _load_json(_resolve(artifact_paths["wave_history"], root=root))
    judge_rows = _load_jsonl(_resolve(artifact_paths["judge_results"], root=root))

    if not isinstance(scope, dict):
        raise ValueError("scope_completeness must be a JSON object")
    if not isinstance(waves, list):
        raise ValueError("wave_history must be a JSON array")
    if not judge_rows:
        raise ValueError("judge_results must contain at least one JSONL record")

    expected_case_id = str(fixture["historical_case_id"])
    judge = next(
        (
            row
            for row in judge_rows
            if str(row.get("case_id") or "") == expected_case_id
        ),
        judge_rows[0],
    )
    category = fixture.get("judge_category", "info_recall")
    category_score = (judge.get("category_scores") or {}).get(category) or {}

    source_count = _list_count(sources, "sources")
    evidence_count = _list_count(evidence, "evidence", "records")
    claim_count = _list_count(claims, "claims")
    entity_count = _list_count(entities, "entities")
    wave_source_delta = sum(int(wave.get("source_delta") or 0) for wave in waves)
    wave_evidence_delta = sum(int(wave.get("evidence_delta") or 0) for wave in waves)
    wave_claim_delta = sum(int(wave.get("claim_delta") or 0) for wave in waves)
    wave_new_entities = sum(
        len((wave.get("scope_completeness") or {}).get("new_entities_by_window") or [])
        for wave in waves
        if isinstance(wave, dict)
    )
    thresholds = fixture["thresholds"]
    allowed_scope_statuses = set(thresholds["allowed_scope_statuses"])
    observed_scope_status = str(scope.get("status") or "unknown")

    observed = {
        "source_count": source_count,
        "evidence_count": evidence_count,
        "claim_count": claim_count,
        "entity_count": entity_count,
        "scope_completeness_status": observed_scope_status,
        "scope_discovered_entity_count": int(
            scope.get("discovered_entity_count") or 0
        ),
        "wave_count": len(waves),
        "wave_source_delta": wave_source_delta,
        "wave_evidence_delta": wave_evidence_delta,
        "wave_claim_delta": wave_claim_delta,
        "wave_new_entities": wave_new_entities,
        "judge_category": category,
        "judge_passed": int(category_score.get("passed") or 0),
        "judge_total": int(category_score.get("total") or 0),
        "judge_score": float(category_score.get("score") or 0.0),
    }
    checks = {
        "upstream_sources_present": source_count
        >= int(thresholds["min_source_count"]),
        "upstream_evidence_present": evidence_count
        >= int(thresholds["min_evidence_count"]),
        "upstream_claims_present": claim_count
        >= int(thresholds["min_claim_count"]),
        "multiple_waves_executed": len(waves) >= int(thresholds["min_wave_count"]),
        "entities_materialized": entity_count
        >= int(thresholds["min_entity_count"]),
        "scope_state_advanced": observed_scope_status in allowed_scope_statuses,
        "wave_entity_progress_observed": wave_new_entities
        >= int(thresholds["min_wave_new_entities"]),
        "information_recall_nonzero": int(category_score.get("passed") or 0)
        >= int(thresholds["min_judge_passed"]),
    }
    diagnostic_checks = {
        key: checks[key]
        for key in (
            "entities_materialized",
            "scope_state_advanced",
            "wave_entity_progress_observed",
            "information_recall_nonzero",
        )
    }
    failed_checks = [key for key, passed in diagnostic_checks.items() if not passed]

    return {
        "schema_version": "historical-entity-pipeline-audit.v1",
        "case_id": fixture["case_id"],
        "historical_case_id": expected_case_id,
        "status": "pass" if not failed_checks else "fail",
        "target_layer": "orchestration_state_pipeline",
        "skill_candidate_eligible": False,
        "skill_candidate_reason": (
            "This frozen run diagnoses cross-wave entity materialization and state "
            "commit behavior; a Skill instruction Candidate cannot replay or repair "
            "that code path by itself."
        ),
        "expected_entity_count": int(thresholds["min_entity_count"]),
        "observed": observed,
        "checks": checks,
        "failed_checks": failed_checks,
        "artifact_paths": artifact_paths,
        "token_usage": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixture_path = _resolve(args.fixture)
    output_path = _resolve(args.output)
    result = audit_entity_pipeline(_load_json(fixture_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
