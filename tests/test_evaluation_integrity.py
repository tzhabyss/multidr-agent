from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.service import ResearchService
from evaluation.io import (
    JsonlRecordConflict,
    append_unique_jsonl,
    load_jsonl,
    load_unique_jsonl,
)
from evaluation.campaign import CampaignTokenLedger
from evaluation.metrics import score_evaluation, score_run_artifacts
from evaluation.runner import EvaluationRunner
from evaluation.replay import build_candidate_replay_observation
from deepresearch.skill_evolution.models import CandidateRecord
from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_identical_jsonl_retry_is_idempotent(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    row = {"case_id": "case-1", "repetition": 1, "answer_sha256": "same"}

    assert append_unique_jsonl(
        path,
        row,
        key_fields=("case_id", "repetition"),
        record_name="prediction",
    )
    assert not append_unique_jsonl(
        path,
        row,
        key_fields=("case_id", "repetition"),
        record_name="prediction",
    )
    assert load_jsonl(path) == [row]


def test_conflicting_prediction_duplicate_is_rejected(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    _jsonl(
        path,
        [
            {"case_id": "case-1", "repetition": 1, "answer_sha256": "first"},
            {"case_id": "case-1", "repetition": 1, "answer_sha256": "second"},
        ],
    )

    with pytest.raises(JsonlRecordConflict, match="case_id='case-1'"):
        load_unique_jsonl(
            path,
            key_fields=("case_id", "repetition"),
            record_name="prediction",
        )


def test_score_evaluation_rejects_conflicting_predictions(tmp_path: Path):
    _jsonl(
        tmp_path / "predictions.jsonl",
        [
            {"case_id": "case-1", "repetition": 1, "status": "failed_system"},
            {"case_id": "case-1", "repetition": 1, "status": "completed_verified"},
        ],
    )

    with pytest.raises(JsonlRecordConflict, match="prediction"):
        score_evaluation(tmp_path)


def test_judge_summary_rejects_conflicting_rows(tmp_path: Path):
    _jsonl(
        tmp_path / "judge_results.jsonl",
        [
            {"case_id": "case-1", "repetition": 1, "answer_sha256": "first"},
            {"case_id": "case-1", "repetition": 1, "answer_sha256": "second"},
        ],
    )

    with pytest.raises(JsonlRecordConflict, match="judge"):
        EvaluationRunner._summarize_judges(tmp_path)


def test_run_score_rejects_conflicting_usage_event(tmp_path: Path):
    details = tmp_path / "details"
    (details / "runtime").mkdir(parents=True)
    (details / "evidence").mkdir(parents=True)
    (details / "manifest.json").write_text(
        json.dumps({"run_id": "run-1", "status": "completed_partial"}),
        encoding="utf-8",
    )
    _jsonl(details / "runtime" / "events.jsonl", [])
    (details / "runtime" / "usage.json").write_text(
        json.dumps(
            [
                {"usage_event_id": "use-1", "input_tokens": 10},
                {"usage_event_id": "use-1", "input_tokens": 20},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(JsonlRecordConflict, match="usage"):
        score_run_artifacts(tmp_path)


def test_campaign_ledger_stops_before_underfunded_sample(tmp_path: Path):
    path = tmp_path / "campaign_token_ledger.json"
    ledger = CampaignTokenLedger(
        path,
        max_total_tokens=1_000,
        max_tokens_per_sample=600,
    )

    assert ledger.reserve("case-1", 1) == 600
    ledger.settle("case-1", 1, consumed_tokens=450)
    assert ledger.reserve("case-2", 1) is None

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["status"] == "exhausted"
    assert persisted["consumed_tokens"] == 450
    assert persisted["remaining_tokens"] == 550
    assert persisted["stop_reason"] == "insufficient_tokens_for_next_sample"


def test_campaign_ledger_resume_and_settlement_are_idempotent(tmp_path: Path):
    path = tmp_path / "campaign_token_ledger.json"
    first = CampaignTokenLedger(
        path,
        max_total_tokens=1_000,
        max_tokens_per_sample=500,
    )
    assert first.reserve("case-1", 1) == 500

    resumed = CampaignTokenLedger(
        path,
        max_total_tokens=1_000,
        max_tokens_per_sample=500,
        resume=True,
    )
    assert resumed.reserve("case-1", 1) == 500
    resumed.settle("case-1", 1, consumed_tokens=100)
    resumed.settle("case-1", 1, consumed_tokens=100)
    with pytest.raises(ValueError, match="Conflicting campaign settlement"):
        resumed.settle("case-1", 1, consumed_tokens=101)


def test_campaign_ledger_records_reservation_breach(tmp_path: Path):
    ledger = CampaignTokenLedger(
        tmp_path / "campaign_token_ledger.json",
        max_total_tokens=1_000,
        max_tokens_per_sample=500,
    )
    ledger.reserve("case-1", 1)
    ledger.settle("case-1", 1, consumed_tokens=600)

    assert ledger.data["status"] == "breached"
    assert ledger.data["stop_reason"] == "sample_reported_usage_exceeded_reservation"
    assert ledger.data["entries"][0]["reservation_exceeded"] is True


def _paired_run(
    root: Path,
    name: str,
    *,
    tokens: int,
    queries: list[str],
    fetch_statuses: list[str],
    judge_score: float,
) -> Path:
    run_dir = root / name
    artifact = root / f"{name}-artifact"
    details = artifact / "details"
    for child in ("runtime", "evidence", "traces", "search"):
        (details / child).mkdir(parents=True, exist_ok=True)
    (details / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "status": "completed_verified",
                "budget": {"max_total_tokens": 1000},
            }
        ),
        encoding="utf-8",
    )
    (details / "runtime" / "usage.json").write_text(
        json.dumps(
            [
                {
                    "usage_event_id": f"use-{name}",
                    "input_tokens": tokens,
                    "output_tokens": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    _jsonl(details / "runtime" / "events.jsonl", [])
    (details / "traces" / "search.json").write_text(
        json.dumps(
            [
                {"query": query, "results": [{"url": f"https://{index}.example"}]}
                for index, query in enumerate(queries)
            ]
        ),
        encoding="utf-8",
    )
    (details / "traces" / "fetch.json").write_text(
        json.dumps([{"status": status} for status in fetch_statuses]),
        encoding="utf-8",
    )
    for filename, value in {
        "sources.json": [
            {
                "source_id": "src-1",
                "canonical_uri": "https://one.example/a",
                "fetched_at": "2026-01-01T00:00:00Z",
                "content_hash": "hash",
            }
        ],
        "source-snapshots.json": [],
        "evidence.json": [],
        "claims.json": [],
        "claim-evidence-links.json": [],
        "citations.json": [],
        "citation-assessment.json": {"passed": True},
    }.items():
        (details / "evidence" / filename).write_text(
            json.dumps(value), encoding="utf-8"
        )
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"dataset_sha256": "same-dataset"}), encoding="utf-8"
    )
    _jsonl(
        run_dir / "predictions.jsonl",
        [
            {
                "case_id": "case-1",
                "repetition": 1,
                "run_id": name,
                "artifact_dir": str(artifact),
                "answer_sha256": f"answer-{name}",
                "citation_gate_passed": True,
                "judge_score": judge_score,
                "judge_category_scores": {
                    "info_recall": {"score": judge_score}
                },
            }
        ],
    )
    return run_dir


def test_paired_evaluations_build_leak_safe_candidate_observation(tmp_path: Path):
    stable = _paired_run(
        tmp_path,
        "stable",
        tokens=500,
        queries=["same query", "same query"],
        fetch_statuses=["failed", "fetched"],
        judge_score=40.0,
    )
    candidate_run = _paired_run(
        tmp_path,
        "candidate",
        tokens=400,
        queries=["first query", "second query"],
        fetch_statuses=["fetched", "fetched"],
        judge_score=60.0,
    )
    candidate = CandidateRecord.model_validate(
        {
            "candidate_id": "candidate-1",
            "source_episode_id": "episode-1",
            "run_id": "source-run",
            "target_skill_id": "research-worker",
            "parent_skill_version": "1.0.0",
            "capability_granularity": "atomic",
            "status": "accept_for_replay",
            "final_patch": {
                "candidate_id": "candidate-1",
                "target_skill_id": "research-worker",
                "parent_skill_version": "1.0.0",
                "capability_granularity": "atomic",
                "trigger_conditions": ["duplicate query"],
                "applicability_preconditions": ["same task"],
                "operations": [
                    {
                        "operation_id": "op-1",
                        "kind": "add",
                        "target_section": "Search",
                        "anchor": "Search",
                        "new_text": "Avoid exact duplicate queries.",
                        "rationale": "reduce waste",
                    }
                ],
                "expected_improvements": ["fewer duplicates"],
                "risks": [],
                "source_episode_ids": ["episode-1"],
            },
        }
    )

    observation = build_candidate_replay_observation(
        stable_run_dir=stable,
        candidate_run_dir=candidate_run,
        candidate=candidate,
        case_id="case-1",
    )

    assert observation.stable_metrics["duplicate_queries"] == 1.0
    assert observation.candidate_metrics["duplicate_queries"] == 0.0
    assert observation.stable_metrics["token_cost"] == 500.0
    assert observation.candidate_metrics["token_cost"] == 400.0
    assert observation.candidate_metrics["judge_info_recall"] == 60.0
    assert observation.hard_checks == {
        "schema_valid": True,
        "permissions_unchanged": True,
        "evidence_gates_preserved": True,
        "external_writes_blocked": True,
    }
    serialized = observation.model_dump_json()
    assert "expected_answer" not in serialized
    assert "atomic_rubrics" not in serialized


@pytest.mark.asyncio
async def test_runner_campaign_budget_stops_additional_samples(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"case_id": "case-1", "question": "Find the verified fact"})
        + "\n",
        encoding="utf-8",
    )
    search = FakeSearchBackend()

    def factory(**kwargs):
        return ResearchService.in_memory(
            llm=FakeLLM(),
            search_backend=search,
            fetcher=FakeFetcher(),
            artifact_root=kwargs["artifact_root"],
        )

    run_dir = await EvaluationRunner(service_factory=factory).run(
        dataset_path=dataset,
        output_root=tmp_path / "evaluations",
        evaluation_id="campaign-stop",
        repetitions=2,
        campaign_token_budget=100,
        max_tokens_per_sample=100,
    )

    assert len(load_jsonl(run_dir / "predictions.jsonl")) == 1
    ledger = json.loads(
        (run_dir / "campaign_token_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "breached"
    assert ledger["entries"][0]["reservation_exceeded"] is True
    report = (run_dir / "evaluation_report.md").read_text(encoding="utf-8")
    assert "Campaign Token Ledger" in report
