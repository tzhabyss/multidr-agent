from __future__ import annotations

import json
from pathlib import Path

from tools.audit_historical_entity_pipeline import audit_entity_pipeline


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_detects_upstream_activity_without_entity_materialization(
    tmp_path: Path,
):
    _write_json(tmp_path / "sources.json", [{"source_id": "s1"}])
    _write_json(tmp_path / "evidence.json", [{"evidence_id": "e1"}])
    _write_json(tmp_path / "claims.json", [{"claim_id": "c1"}])
    _write_json(tmp_path / "entities.json", [])
    _write_json(
        tmp_path / "scope.json",
        {"status": "unknown", "discovered_entity_count": 0},
    )
    _write_json(
        tmp_path / "waves.json",
        [
            {
                "source_delta": 1,
                "evidence_delta": 2,
                "claim_delta": 1,
                "scope_completeness": {"new_entities_by_window": []},
            },
            {
                "source_delta": 1,
                "evidence_delta": 1,
                "claim_delta": 1,
                "scope_completeness": {"new_entities_by_window": []},
            },
        ],
    )
    (tmp_path / "judge.jsonl").write_text(
        json.dumps(
            {
                "case_id": "historical-1",
                "category_scores": {
                    "info_recall": {"passed": 0, "total": 3, "score": 0.0}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture = {
        "case_id": "audit-1",
        "historical_case_id": "historical-1",
        "judge_category": "info_recall",
        "artifact_paths": {
            "sources": "sources.json",
            "evidence": "evidence.json",
            "claims": "claims.json",
            "entities": "entities.json",
            "scope_completeness": "scope.json",
            "wave_history": "waves.json",
            "judge_results": "judge.jsonl",
        },
        "thresholds": {
            "min_source_count": 1,
            "min_evidence_count": 1,
            "min_claim_count": 1,
            "min_wave_count": 2,
            "min_entity_count": 3,
            "allowed_scope_statuses": ["stabilizing", "provisionally_complete"],
            "min_wave_new_entities": 1,
            "min_judge_passed": 1,
        },
    }

    result = audit_entity_pipeline(fixture, root=tmp_path)

    assert result["status"] == "fail"
    assert result["observed"]["source_count"] == 1
    assert result["observed"]["evidence_count"] == 1
    assert result["observed"]["claim_count"] == 1
    assert result["observed"]["entity_count"] == 0
    assert result["checks"]["upstream_sources_present"] is True
    assert result["failed_checks"] == [
        "entities_materialized",
        "scope_state_advanced",
        "wave_entity_progress_observed",
        "information_recall_nonzero",
    ]
    assert result["skill_candidate_eligible"] is False


def test_audit_passes_when_entities_and_scope_progress(tmp_path: Path):
    _write_json(tmp_path / "sources.json", [{"source_id": "s1"}])
    _write_json(tmp_path / "evidence.json", [{"evidence_id": "e1"}])
    _write_json(tmp_path / "claims.json", [{"claim_id": "c1"}])
    _write_json(
        tmp_path / "entities.json",
        [{"entity_id": "a"}, {"entity_id": "b"}],
    )
    _write_json(
        tmp_path / "scope.json",
        {"status": "provisionally_complete", "discovered_entity_count": 2},
    )
    _write_json(
        tmp_path / "waves.json",
        [
            {
                "source_delta": 1,
                "evidence_delta": 1,
                "claim_delta": 1,
                "scope_completeness": {"new_entities_by_window": ["a", "b"]},
            }
        ],
    )
    (tmp_path / "judge.jsonl").write_text(
        json.dumps(
            {
                "case_id": "historical-2",
                "category_scores": {
                    "info_recall": {"passed": 2, "total": 2, "score": 100.0}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture = {
        "case_id": "audit-2",
        "historical_case_id": "historical-2",
        "judge_category": "info_recall",
        "artifact_paths": {
            "sources": "sources.json",
            "evidence": "evidence.json",
            "claims": "claims.json",
            "entities": "entities.json",
            "scope_completeness": "scope.json",
            "wave_history": "waves.json",
            "judge_results": "judge.jsonl",
        },
        "thresholds": {
            "min_source_count": 1,
            "min_evidence_count": 1,
            "min_claim_count": 1,
            "min_wave_count": 1,
            "min_entity_count": 2,
            "allowed_scope_statuses": ["provisionally_complete"],
            "min_wave_new_entities": 1,
            "min_judge_passed": 1,
        },
    }

    result = audit_entity_pipeline(fixture, root=tmp_path)

    assert result["status"] == "pass"
    assert result["failed_checks"] == []
    assert all(result["checks"].values())
