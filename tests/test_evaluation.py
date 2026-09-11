from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
from pathlib import Path

import pytest

from deepresearch.service import ResearchService
from deepresearch.llm import LLMResponse
from deepresearch.date_constraints import source_within_date_boundary
from deepresearch.schemas import SearchHit
from deepresearch.model_routing import ModelRouter
from evaluation.drb2 import normalize_drb2_record
from evaluation import cli as evaluation_cli
from evaluation.io import load_jsonl
from evaluation.judge import RubricJudge
from evaluation.metrics import score_run_artifacts
from evaluation.runner import EvaluationRunner, _runtime_config_for_evaluation
from evaluation.schemas import EvaluationCase

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


@pytest.mark.asyncio
async def test_drb2_cli_reuses_cached_selection_without_downloading(
    tmp_path: Path, monkeypatch
):
    selection = json.dumps(
        {"case_ids": ["task-cache"], "limit": None}, sort_keys=True
    ).encode("utf-8")
    label = hashlib.sha256(selection).hexdigest()[:10]
    dataset = tmp_path / ".datasets" / f"drb2-{label}.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            {
                "case_id": "task-cache",
                "question": "cached benchmark question",
                "mode": "standard",
                "profile_id": "drb_evaluation",
                "constraints": {},
                "atomic_rubrics": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def unexpected_download(*args, **kwargs):
        raise AssertionError("cached DRB2 selection must not be downloaded again")

    captured: dict[str, object] = {}

    async def fake_run(self, **kwargs):
        captured.update(kwargs)
        return tmp_path / "result"

    monkeypatch.setattr(evaluation_cli, "prepare_drb2_dataset", unexpected_download)
    monkeypatch.setattr(evaluation_cli.EvaluationRunner, "run", fake_run)
    args = argparse.Namespace(
        dataset=None,
        benchmark="drb2",
        output_root=tmp_path,
        evaluation_id="cache-test",
        repetitions=1,
        seed=42,
        track="general_research",
        variant="p1_full",
        dataset_revision=None,
        resume=False,
        case_id=["task-cache"],
        limit=None,
        mode="standard",
        profile=None,
        judge=True,
        judge_tier="default",
        search_provider=[],
        search_provider_mode="fanout",
    )

    await evaluation_cli._run(args)

    assert captured["dataset_path"] == dataset
    assert captured["track"] == "drb2"


class FakeRubricLLM:
    model_id = "fake-medium-model"

    async def complete_json(self, system: str, user: str, *, max_tokens: int):
        payload = json.loads(user)
        return {
            "decisions": [
                {
                    "rubric_id": item["rubric_id"],
                    "satisfied": item["rubric_id"].endswith("001"),
                    "evidence": "42" if item["rubric_id"].endswith("001") else "",
                    "reason": "fixture decision",
                }
                for item in payload["rubrics"]
            ]
        }, LLMResponse("{}", input_tokens=10, output_tokens=5)


class EvaluationFakeLLM(FakeLLM):
    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        if "strict benchmark evaluator" in system:
            payload = json.loads(user)
            return {
                "decisions": [
                    {
                        "rubric_id": item["rubric_id"],
                        "satisfied": True,
                        "evidence": "verified fact",
                        "reason": "fixture answer satisfies the criterion",
                    }
                    for item in payload["rubrics"]
                ]
            }, LLMResponse("{}", input_tokens=11, output_tokens=6)
        return await super().complete_json(system, user, max_tokens=max_tokens)


def test_evaluation_runtime_enables_default_fanout_search_mcps():
    evaluation = _runtime_config_for_evaluation({"drb_evaluation"})
    assert evaluation.mcp.servers["deepseek-web-search"].enabled_env is None
    assert evaluation.mcp.servers["anysearch"].enabled_env is None
    profile = evaluation.profile_config("drb_evaluation")
    assert profile.skill_fallbacks["web-search"] == [
        "deepseek-web-search",
        "anysearch-search",
    ]
    assert profile.skill_provider_modes["web-search"] == "fanout"

    general = _runtime_config_for_evaluation({"general_research"})
    assert general.mcp.servers["deepseek-web-search"].enabled_env is None
    general_profile = general.profile_config("general_research")
    assert general_profile.skill_fallbacks["web-search"] == [
        "deepseek-web-search",
        "anysearch-search",
    ]
    assert general_profile.skill_provider_modes["web-search"] == "fanout"


def test_evaluation_runtime_can_select_anysearch_or_provider_fanout():
    anysearch = _runtime_config_for_evaluation(
        {"drb_evaluation"},
        search_providers=["anysearch-search"],
    )
    profile = anysearch.profile_config("drb_evaluation")
    assert profile.skill_fallbacks["web-search"] == ["anysearch-search"]
    assert anysearch.mcp.servers["anysearch"].enabled_env is None
    assert anysearch.mcp.servers["deepseek-web-search"].enabled_env is None

    combined = _runtime_config_for_evaluation(
        {"drb_evaluation"},
        search_providers=["deepseek-web-search", "anysearch-search"],
        search_provider_mode="fanout",
    )
    combined_profile = combined.profile_config("drb_evaluation")
    assert combined_profile.skill_provider_modes["web-search"] == "fanout"
    assert combined.mcp.servers["deepseek-web-search"].enabled_env is None
    assert combined.mcp.servers["anysearch"].enabled_env is None


def test_drb2_adapter_hides_rubrics_and_enforces_exact_blocked_urls():
    normalized = normalize_drb2_record(
        {
            "id": "task-test",
            "idx": 1,
            "language": "zh",
            "content": {
                "task": "公开问题",
                "rubric": {
                    "info_recall": ["SECRET_RUBRIC"],
                    "analysis": [],
                    "presentation": [],
                },
                "blocked": {"urls": ["https://blocked.example/article"]},
            },
        }
    )
    case = EvaluationCase.model_validate(normalized)
    request = case.agent_request("eval-task-test")
    dumped = request.model_dump_json()
    assert request.profile_id == "drb_evaluation"
    assert request.constraints.blocked_urls == ["https://blocked.example/article"]
    assert "SECRET_RUBRIC" not in dumped


def test_drb2_adapter_extracts_visible_month_cutoff_and_filters_future_sources():
    normalized = normalize_drb2_record(
        {
            "id": "dated-task",
            "language": "en",
            "content": {
                "task": "Compare the available systems using evidence up to April 2024.",
                "rubric": {},
                "blocked": {},
            },
        }
    )
    assert normalized["constraints"]["date_boundary"] == "2024-04-30"

    metadata_hit = SearchHit(
        title="Later analysis",
        url="https://example.com/analysis",
        provider="fixture",
        published_at="2025-01-02",
    )
    url_hit = SearchHit(
        title="November update",
        url="https://example.com/2024/11/update",
        provider="fixture",
    )
    unknown_hit = SearchHit(
        title="A history of 2026 forecasts",
        url="https://example.com/history",
        provider="fixture",
    )

    assert source_within_date_boundary(metadata_hit, "2024-04-30")[0] is False
    assert source_within_date_boundary(url_hit, "2024-04-30")[0] is False
    assert source_within_date_boundary(unknown_hit, "2024-04-30")[0] is True


@pytest.mark.asyncio
async def test_rubric_judge_returns_auditable_category_scores():
    case = EvaluationCase.model_validate(
        {
            "case_id": "judge-case",
            "question": "What is the answer?",
            "atomic_rubrics": [
                {
                    "rubric_id": "info_recall-001",
                    "category": "info_recall",
                    "criterion": "States 42",
                },
                {
                    "rubric_id": "analysis-002",
                    "category": "analysis",
                    "criterion": "Explains the cause",
                },
            ],
        }
    )
    result = await RubricJudge(FakeRubricLLM()).judge(case, "The answer is 42.")
    assert result["judge_model"] == "fake-medium-model"
    assert result["total"] == len(case.atomic_rubrics)
    assert result["passed"] == 1
    assert result["total"] == 2
    assert result["overall_score"] == 50.0
    assert result["decisions"][0]["evidence"] == "42"


@pytest.mark.asyncio
async def test_rubric_judge_times_out_a_stalled_batch():
    class SlowRubricLLM:
        model_id = "slow-judge"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            await asyncio.sleep(1)
            return {"decisions": []}, LLMResponse("{}", input_tokens=0, output_tokens=0)

    case = EvaluationCase.model_validate(
        {
            "case_id": "slow-judge-case",
            "question": "What is the answer?",
            "atomic_rubrics": [
                {"rubric_id": "info_recall-001", "category": "info_recall", "criterion": "States 42"}
            ],
        }
    )
    with pytest.raises(asyncio.TimeoutError):
        await RubricJudge(SlowRubricLLM(), batch_timeout_seconds=0.01).judge(case, "answer")


@pytest.mark.asyncio
async def test_rubric_judge_retries_a_timed_out_batch():
    class FlakyRubricLLM:
        model_id = "flaky-judge"

        def __init__(self):
            self.calls = 0

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(1)
            rubric_id = json.loads(user)["rubrics"][0]["rubric_id"]
            return {
                "decisions": [{"rubric_id": rubric_id, "satisfied": True}]
            }, LLMResponse("{}", input_tokens=3, output_tokens=2)

    llm = FlakyRubricLLM()
    case = EvaluationCase.model_validate(
        {
            "case_id": "flaky-judge-case",
            "question": "What is the answer?",
            "atomic_rubrics": [
                {"rubric_id": "recall-001", "category": "recall", "criterion": "States 42"}
            ],
        }
    )

    result = await RubricJudge(
        llm,
        batch_timeout_seconds=0.01,
        max_attempts=2,
        retry_backoff_seconds=0,
    ).judge(case, "42")

    assert llm.calls == 2
    assert result["passed"] == result["total"] == 1


@pytest.mark.asyncio
async def test_rubric_judge_splits_repeatedly_timed_out_multi_rubric_batch():
    class SplitRubricLLM:
        model_id = "split-judge"

        async def complete_json(self, system: str, user: str, *, max_tokens: int):
            rubrics = json.loads(user)["rubrics"]
            if len(rubrics) > 1:
                await asyncio.sleep(1)
            return {
                "decisions": [
                    {"rubric_id": item["rubric_id"], "satisfied": True}
                    for item in rubrics
                ]
            }, LLMResponse("{}", input_tokens=1, output_tokens=1)

    case = EvaluationCase.model_validate(
        {
            "case_id": "split-judge-case",
            "question": "Check all requirements",
            "atomic_rubrics": [
                {"rubric_id": f"recall-{index}", "category": "recall", "criterion": str(index)}
                for index in range(4)
            ],
        }
    )

    result = await RubricJudge(
        SplitRubricLLM(),
        batch_size=4,
        batch_timeout_seconds=0.01,
        max_attempts=1,
    ).judge(case, "complete")

    assert result["passed"] == result["total"] == 4
    assert {item["rubric_id"] for item in result["decisions"]} == {
        "recall-0", "recall-1", "recall-2", "recall-3"
    }


def test_evaluation_case_never_forwards_reference_fields():
    case = EvaluationCase.model_validate(
        {
            "case_id": "reference-isolation",
            "question": "What is public?",
            "mode": "deep_search",
            "expected_answer": "SECRET_CANARY",
            "reference_source_ids": ["gold-secret-source"],
            "atomic_rubrics": [{"rubric_id": "secret-rubric"}],
            "hidden_reference": "ANOTHER_CANARY",
        }
    )
    dumped = case.agent_request("eval-reference-isolation").model_dump_json()
    assert "SECRET_CANARY" not in dumped
    assert "gold-secret-source" not in dumped
    assert "secret-rubric" not in dumped
    assert "ANOTHER_CANARY" not in dumped


@pytest.mark.asyncio
async def test_evaluation_runner_persists_scores_report_and_resumes(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "question": "What is the verified fact?",
                "mode": "deep_search",
                "expected_answer": "reference-only",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first_search = FakeSearchBackend()

    def first_factory(**kwargs):
        return ResearchService.in_memory(
            llm=FakeLLM(),
            search_backend=first_search,
            fetcher=FakeFetcher(),
            artifact_root=kwargs["artifact_root"],
        )

    runner = EvaluationRunner(service_factory=first_factory)
    run_dir = await runner.run(
        dataset_path=dataset,
        output_root=tmp_path / "evaluations",
        evaluation_id="resume-test",
        repetitions=2,
        seed=42,
    )

    predictions = load_jsonl(run_dir / "predictions.jsonl")
    scores = load_jsonl(run_dir / "deterministic_scores.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(predictions) == 2
    assert len(scores) == 2
    assert len(first_search.calls) == 2
    assert summary["verified_run_rate"] == 1.0
    assert summary["citation_grounding_precision"] == 1.0
    assert summary["quote_match_rate"] == 1.0
    assert summary["budget_compliance_rate"] == 1.0
    assert summary["terminal_event_exactly_once_rate"] == 1.0
    assert (run_dir / "evaluation_report.md").exists()

    second_search = FakeSearchBackend()

    def second_factory(**kwargs):
        return ResearchService.in_memory(
            llm=FakeLLM(),
            search_backend=second_search,
            fetcher=FakeFetcher(),
            artifact_root=kwargs["artifact_root"],
        )

    resumed = EvaluationRunner(service_factory=second_factory)
    await resumed.run(
        dataset_path=dataset,
        output_root=tmp_path / "evaluations",
        evaluation_id="resume-test",
        repetitions=2,
        seed=42,
        resume=True,
    )
    assert second_search.calls == []
    assert len(load_jsonl(run_dir / "predictions.jsonl")) == 2


@pytest.mark.asyncio
async def test_evaluation_runner_uses_injected_medium_tier_judge(tmp_path: Path):
    dataset = tmp_path / "judge-dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "judge-runner-case",
                "question": "What is the verified fact?",
                "profile_id": "drb_evaluation",
                "atomic_rubrics": [
                    {
                        "rubric_id": "info_recall-001",
                        "category": "info_recall",
                        "criterion": "Includes the verified fact",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def factory(**kwargs):
        return ResearchService.in_memory(
            llm=EvaluationFakeLLM(),
            search_backend=FakeSearchBackend(),
            fetcher=FakeFetcher(),
            artifact_root=kwargs["artifact_root"],
        )

    run_dir = await EvaluationRunner(service_factory=factory).run(
        dataset_path=dataset,
        output_root=tmp_path / "evaluations",
        evaluation_id="judge-runner",
        judge=True,
        judge_tier="default",
    )
    judge_summary = json.loads(
        (run_dir / "judge_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert judge_summary["overall_score"] == 100.0
    assert manifest["judge_tier"] == "default"
    assert manifest["judge_model"] == "fake-research-model"
    assert len(load_jsonl(run_dir / "judge_results.jsonl")) == 1


@pytest.mark.asyncio
async def test_finalize_only_rebuilds_report_without_new_search(tmp_path: Path):
    search = FakeSearchBackend()
    llm = FakeLLM()
    service = ResearchService.in_memory(
        llm=llm,
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        first = await service.run(
            {
                "question": "What is the verified fact?",
                "mode": "quick",
                "run_id": "finalize-only-fixture",
            }
        )
        search_count = len(search.calls)
        finalized = await service.finalize_only(first.run_id)
        finalized_again = await service.finalize_only(first.run_id)

    assert len(search.calls) == search_count
    assert finalized.output_path == first.output_path
    assert finalized_again.output_path == first.output_path
    assert Path(finalized.output_path).exists()
    events = load_jsonl(
        Path(finalized.artifact_dir) / "details" / "runtime" / "events.jsonl"
    )
    assert sum(item.get("event_type") == "run_terminal" for item in events) == 1
    usage = json.loads(
        (Path(finalized.artifact_dir) / "details" / "runtime" / "usage.json").read_text(
            encoding="utf-8"
        )
    )
    usage_ids = [item["usage_event_id"] for item in usage]
    assert len(usage_ids) == len(set(usage_ids))


@pytest.mark.asyncio
async def test_judge_only_scores_frozen_prediction_without_generation(tmp_path: Path):
    dataset = tmp_path / "judge-only-dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "judge-only-case",
                "question": "What is the verified fact?",
                "profile_id": "drb_evaluation",
                "atomic_rubrics": [
                    {
                        "rubric_id": "info_recall-001",
                        "category": "info_recall",
                        "criterion": "Includes the verified fact",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    search = FakeSearchBackend()

    def factory(**kwargs):
        return ResearchService.in_memory(
            llm=EvaluationFakeLLM(),
            search_backend=search,
            fetcher=FakeFetcher(),
            artifact_root=kwargs["artifact_root"],
        )

    run_dir = await EvaluationRunner(service_factory=factory).run(
        dataset_path=dataset,
        output_root=tmp_path / "evaluations",
        evaluation_id="judge-only-eval",
        judge=False,
    )
    search_count = len(search.calls)
    judge_runner = EvaluationRunner(
        model_router_factory=lambda config: ModelRouter(
            config, injected_llm=EvaluationFakeLLM()
        )
    )
    await judge_runner.judge_only(run_dir, judge_tier="default")
    await judge_runner.judge_only(run_dir, judge_tier="default")

    assert len(search.calls) == search_count
    predictions = load_jsonl(run_dir / "predictions.jsonl")
    assert predictions[0]["judge_score"] == 100.0
    assert predictions[0]["judge_error"] is None
    assert len(load_jsonl(run_dir / "judge_results.jsonl")) == 1


@pytest.mark.asyncio
async def test_research_hard_filters_search_hits_after_date_boundary(tmp_path: Path):
    class DatedSearch:
        name = "dated-search"

        async def search(self, query: str, max_results: int = 5):
            return [
                SearchHit(
                    title="Future source",
                    url="https://future.example/2025/01/article",
                    provider=self.name,
                    rank=1,
                    published_at="2025-01-02",
                ),
                SearchHit(
                    title="In-bound source",
                    url="https://past.example/2024/03/article",
                    provider=self.name,
                    rank=2,
                    published_at="2024-03-01",
                ),
            ]

    fetcher = FakeFetcher()
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=DatedSearch(),
        fetcher=fetcher,
        artifact_root=tmp_path / "dated-artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "Use evidence up to April 2024",
                "mode": "quick",
                "profile_id": "drb_evaluation",
                "constraints": {"date_boundary": "2024-04-30"},
            }
        )

    assert fetcher.calls
    assert all("future.example" not in url for url in fetcher.calls)
    assert all(item.published_at == "2024-03-01" for item in result.sources)


@pytest.mark.asyncio
async def test_run_artifact_metrics_are_grounded_in_typed_ledger(tmp_path: Path):
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
    )
    async with service:
        result = await service.run(
            {
                "question": "Score this run",
                "mode": "deep_search",
                "run_id": "metric-test-run",
            }
        )

    score = score_run_artifacts(result.artifact_dir)
    assert score["verified"]
    assert score["citation_gate_passed"]
    assert score["citation_grounding_precision"] == 1.0
    assert score["quote_match_rate"] == 1.0
    assert score["unsupported_claim_rate"] == 0.0
    assert score["high_impact_claim_coverage"] == 1.0
    assert score["budget_compliant"]
    assert score["terminal_event_exactly_once"]
