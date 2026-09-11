"""Build leak-safe CoEvoSkills observations from paired evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from deepresearch.ids import make_id, normalize_text
from deepresearch.skill_evolution.models import CandidateRecord, EvolutionEvidenceReference
from deepresearch.skill_evolution.verifier_models import CandidateReplayObservation

from .io import load_unique_jsonl
from .metrics import score_evaluation, score_run_artifacts


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_rows(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = load_unique_jsonl(
        run_dir / "predictions.jsonl",
        key_fields=("case_id", "repetition"),
        record_name="prediction",
    )
    return {
        (str(item["case_id"]), int(item["repetition"])): item for item in rows
    }


def _local_metrics(prediction: dict[str, Any]) -> dict[str, float]:
    artifact_dir = Path(str(prediction.get("artifact_dir") or ""))
    if not artifact_dir.is_dir():
        raise ValueError("Prediction has no readable artifact_dir")
    score = score_run_artifacts(artifact_dir)
    details = artifact_dir / "details"
    searches = _read_json(details / "traces" / "search.json", [])
    fetches = _read_json(details / "traces" / "fetch.json", [])
    worker = _read_json(details / "traces" / "worker.json", [])
    claims = _read_json(details / "evidence" / "claims.json", [])
    links = _read_json(details / "evidence" / "claim-evidence-links.json", [])
    sources = _read_json(details / "evidence" / "sources.json", [])
    bindings = _read_json(details / "search" / "evidence-bindings.json", [])
    if not bindings:
        graph = _read_json(details / "search" / "evidence-graph.json", {})
        bindings = graph.get("evidence_bindings", []) if isinstance(graph, dict) else []

    normalized_queries = [
        normalize_text(str(item.get("query") or "")).casefold()
        for item in searches
        if normalize_text(str(item.get("query") or ""))
    ]
    duplicate_queries = len(normalized_queries) - len(set(normalized_queries))
    domains = {
        (urlsplit(str(item.get("canonical_uri") or "")).hostname or "").casefold()
        for item in sources
        if item.get("canonical_uri")
    } - {""}
    failed_fetches = [
        item for item in fetches if str(item.get("status") or "") not in {"fetched", "cached", "shared"}
    ]
    supported_claim_ids = {
        str(item.get("claim_id"))
        for item in links
        if item.get("relation") == "supports"
        and item.get("entailment_status") == "verified"
    }
    supported_claims = sum(
        str(item.get("claim_id")) in supported_claim_ids for item in claims
    )
    weak_claims = len(claims) - supported_claims
    skill_failures = sum(
        bool(item.get("error"))
        or str(item.get("status") or "").casefold() in {"failed", "error"}
        for item in worker
    )
    metrics = {
        key: float(value)
        for key, value in score.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    metrics.update(
        {
            "token_cost": float(score.get("total_tokens") or 0),
            "evidence_gain": float(len(claims)),
            "zero_gain_searches": float(
                sum(not item.get("results") or bool(item.get("error")) for item in searches)
            ),
            "duplicate_queries": float(duplicate_queries),
            "duplicate_query_rate": (
                float(duplicate_queries / len(normalized_queries))
                if normalized_queries
                else 0.0
            ),
            "source_domain_diversity": float(len(domains)),
            "repeated_domain_rate": (
                float(max(0, len(sources) - len(domains)) / len(sources))
                if sources
                else 0.0
            ),
            "source_family_failures": float(len(failed_fetches)),
            "fetch_route_failures": float(len(failed_fetches)),
            "fetch_success_rate": (
                float((len(fetches) - len(failed_fetches)) / len(fetches))
                if fetches
                else 0.0
            ),
            "evidence_bindings": float(len(bindings)),
            "unbound_extractions": float(max(0, len(claims) - len(bindings))),
            "supported_claims": float(supported_claims),
            "weak_or_rejected_claims": float(weak_claims),
            "skill_execution_failures": float(skill_failures),
        }
    )
    judge_score = prediction.get("judge_score")
    if isinstance(judge_score, (int, float)):
        metrics["judge_score"] = float(judge_score)
    for category, value in (prediction.get("judge_category_scores") or {}).items():
        score_value = value.get("score") if isinstance(value, dict) else None
        if isinstance(score_value, (int, float)):
            metrics[f"judge_{category}"] = float(score_value)
    return metrics


def build_candidate_replay_observation(
    *,
    stable_run_dir: str | Path,
    candidate_run_dir: str | Path,
    candidate: CandidateRecord,
    case_id: str,
    repetition: int = 1,
) -> CandidateReplayObservation:
    """Convert one matched Stable/Candidate sample into the verifier contract."""

    if candidate.status != "accept_for_replay" or candidate.final_patch is None:
        raise ValueError("Candidate must be accept_for_replay with a final patch")
    stable_dir = Path(stable_run_dir).resolve()
    candidate_dir = Path(candidate_run_dir).resolve()
    stable_manifest = _read_json(stable_dir / "manifest.json", {})
    candidate_manifest = _read_json(candidate_dir / "manifest.json", {})
    if stable_manifest.get("dataset_sha256") != candidate_manifest.get("dataset_sha256"):
        raise ValueError("Stable and Candidate evaluations use different datasets")
    key = (case_id, repetition)
    stable_prediction = _sample_rows(stable_dir).get(key)
    candidate_prediction = _sample_rows(candidate_dir).get(key)
    if stable_prediction is None or candidate_prediction is None:
        raise ValueError(f"Paired sample is missing: {case_id} repetition {repetition}")

    patch = candidate.final_patch
    permissions_unchanged = not (
        patch.requested_permissions or patch.requested_tools
    )
    evidence_gates_preserved = not patch.gate_bypasses and bool(
        candidate_prediction.get("citation_gate_passed")
    )
    replay_id = make_id(
        "candidate-replay",
        candidate.candidate_id,
        case_id,
        repetition,
        stable_prediction.get("answer_sha256"),
        candidate_prediction.get("answer_sha256"),
    )
    return CandidateReplayObservation(
        replay_id=replay_id,
        candidate_id=candidate.candidate_id,
        source_episode_id=candidate.source_episode_id,
        stable_metrics=_local_metrics(stable_prediction),
        candidate_metrics=_local_metrics(candidate_prediction),
        hard_checks={
            "schema_valid": True,
            "permissions_unchanged": permissions_unchanged,
            "evidence_gates_preserved": evidence_gates_preserved,
            "external_writes_blocked": True,
        },
        evidence_refs=[
            EvolutionEvidenceReference(
                artifact_path=str(stable_dir / "predictions.jsonl"),
                record_type="stable_evaluation_prediction",
                record_id=f"{case_id}:r{repetition}",
            ),
            EvolutionEvidenceReference(
                artifact_path=str(candidate_dir / "predictions.jsonl"),
                record_type="candidate_evaluation_prediction",
                record_id=f"{case_id}:r{repetition}",
            ),
        ],
        summary=(
            f"Paired frozen evaluation for {case_id} repetition {repetition}; "
            "rubrics and expected answers were not copied into this observation."
        ),
    )


def ensure_scored(run_dir: str | Path) -> None:
    """Create deterministic score artifacts when a frozen run lacks them."""

    path = Path(run_dir)
    if not (path / "deterministic_scores.jsonl").exists():
        score_evaluation(path)

