"""JSONL evaluation runner with per-sample persistence and resume."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from deepresearch.ids import stable_hash
from deepresearch.runtime_config import RuntimeConfig, load_runtime_config
from deepresearch.service import ResearchService, RunNotFoundError
from deepresearch.model_routing import ModelRouter

from .io import (
    append_unique_jsonl,
    load_cases,
    load_unique_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .metrics import score_evaluation
from .report import render_report
from .judge import RubricJudge
from .campaign import CampaignTokenLedger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_id(case_id: str, repetition: int) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", case_id).strip("-")[:40] or "case"
    return f"eval-{label}-r{repetition}-{stable_hash(case_id, repetition)[:8]}"


def _runtime_config_for_evaluation(
    profile_ids: set[str],
    *,
    search_providers: list[str] | None = None,
    search_provider_mode: str = "fanout",
) -> RuntimeConfig:
    """Apply an evaluation provider selection and ensure its MCPs are enabled."""

    config = load_runtime_config()
    if search_providers:
        profiles = dict(config.agent_profiles)
        for profile_id in profile_ids:
            profile = config.profile_config(profile_id)
            fallbacks = dict(profile.skill_fallbacks)
            fallbacks["web-search"] = list(dict.fromkeys(search_providers))
            provider_modes = dict(profile.skill_provider_modes)
            provider_modes["web-search"] = search_provider_mode
            profiles[profile_id] = profile.model_copy(
                update={
                    "skill_fallbacks": fallbacks,
                    "skill_provider_modes": provider_modes,
                }
            )
        config = config.model_copy(update={"agent_profiles": profiles})
    selected_skill_ids = {
        skill_id
        for profile_id in profile_ids
        for skill_id in config.profile_config(profile_id).skill_fallbacks.get(
            "web-search", []
        )
    }
    servers = dict(config.mcp.servers)
    changed = False
    for server_id, server in list(servers.items()):
        exposed_skill_ids = {
            policy.skill_id
            for policy in server.tools.values()
            if policy.skill_id
        }
        if selected_skill_ids & exposed_skill_ids:
            servers[server_id] = server.model_copy(update={"enabled_env": None})
            changed = True
    if not changed:
        return config
    return config.model_copy(
        update={"mcp": config.mcp.model_copy(update={"servers": servers})}
    )


class EvaluationRunner:
    # A rubric-heavy case may require many sequential JSON batches. Keep the
    # per-batch guard and whole-case guard separate so one slow request does
    # not make the whole judge run fail prematurely.
    JUDGE_BATCH_TIMEOUT_SECONDS = 240.0
    # Generation now reserves a long Writer slot after the expanded research
    # budget. Judge gets up to twelve additional minutes after generation.
    JUDGE_TIMEOUT_SECONDS = 720.0
    # Leave room for the expanded research wall time plus the Judge pass.
    CASE_TIMEOUT_SECONDS = 7800.0
    JUDGE_MAX_ATTEMPTS = 2

    def __init__(
        self,
        service_factory: Callable[..., Any] = ResearchService,
        model_router_factory: Callable[[RuntimeConfig], ModelRouter] = ModelRouter,
    ):
        self.service_factory = service_factory
        self.model_router_factory = model_router_factory

    async def run(
        self,
        *,
        dataset_path: str | Path,
        output_root: str | Path = "artifacts/evaluations",
        evaluation_id: str | None = None,
        repetitions: int = 1,
        seed: int = 42,
        resume: bool = False,
        track: str = "general_research",
        variant: str = "p1_full",
        dataset_revision: str | None = None,
        mode: str | None = None,
        profile_id: str | None = None,
        judge: bool = False,
        judge_tier: str = "default",
        search_providers: list[str] | None = None,
        search_provider_mode: str = "fanout",
        campaign_token_budget: int | None = None,
        max_tokens_per_sample: int | None = None,
    ) -> Path:
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if campaign_token_budget is not None and max_tokens_per_sample is None:
            raise ValueError(
                "max_tokens_per_sample is required with campaign_token_budget"
            )
        if campaign_token_budget is None and max_tokens_per_sample is not None:
            raise ValueError(
                "campaign_token_budget is required with max_tokens_per_sample"
            )
        dataset_path = Path(dataset_path).resolve()
        cases = load_cases(dataset_path)
        evaluation_id = evaluation_id or (
            datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ-")
            + sha256_file(dataset_path)[:8]
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]+", evaluation_id):
            raise ValueError("evaluation_id may contain only letters, digits, '-' and '_'")
        run_dir = (Path(output_root).resolve() / evaluation_id)
        predictions_path = run_dir / "predictions.jsonl"
        if predictions_path.exists() and predictions_path.stat().st_size and not resume:
            raise FileExistsError("Evaluation already has predictions; pass resume=True")
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.json"
        manifest = {
            "schema_version": "1.0",
            "evaluation_id": evaluation_id,
            "created_at": _utc_now(),
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_revision": dataset_revision,
            "sample_count": len(cases),
            "repetitions": repetitions,
            "seed": seed,
            "track": track,
            "variant": variant,
            "generator_model": None,
            "generator_routing": None,
            "judge_model": None,
            "judge_tier": judge_tier if judge else None,
            "search_providers": search_providers,
            "search_provider_mode": search_provider_mode,
            "reference_fields_visible_to_agent": False,
            "campaign_token_budget": campaign_token_budget,
            "max_tokens_per_sample": max_tokens_per_sample,
        }
        if manifest_path.exists() and resume:
            existing_manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            immutable = (
                "dataset_sha256",
                "repetitions",
                "seed",
                "track",
                "variant",
                "judge_tier",
                "search_providers",
                "search_provider_mode",
                "campaign_token_budget",
                "max_tokens_per_sample",
            )
            mismatches = [key for key in immutable if existing_manifest.get(key) != manifest.get(key)]
            if mismatches:
                raise ValueError("Resume configuration mismatch: " + ", ".join(mismatches))
            manifest = existing_manifest
        else:
            write_json(manifest_path, manifest)

        sample_keys = [
            {"case_id": case.case_id, "repetition": repetition}
            for case in cases
            for repetition in range(1, repetitions + 1)
        ]
        write_json(run_dir / "sample_ids.json", sample_keys)
        prediction_rows = load_unique_jsonl(
            predictions_path,
            key_fields=("case_id", "repetition"),
            record_name="prediction",
        )
        if resume:
            retryable_rows = [
                item for item in prediction_rows if item.get("status") == "failed_system"
            ]
            if retryable_rows:
                prediction_rows = [
                    item for item in prediction_rows if item.get("status") != "failed_system"
                ]
                write_jsonl(predictions_path, prediction_rows)
        completed = {
            (item.get("case_id"), item.get("repetition"))
            for item in prediction_rows
        }
        campaign = (
            CampaignTokenLedger(
                run_dir / "campaign_token_ledger.json",
                max_total_tokens=int(campaign_token_budget),
                max_tokens_per_sample=int(max_tokens_per_sample),
                resume=resume,
            )
            if campaign_token_budget is not None
            else None
        )

        samples_root = run_dir / "samples"
        checkpoint_path = run_dir / "checkpoints.sqlite"
        judge_results_path = run_dir / "judge_results.jsonl"
        service_kwargs: dict[str, Any] = {
            "artifact_root": samples_root,
            "checkpoint_path": checkpoint_path,
        }
        if self.service_factory is ResearchService:
            service_kwargs["runtime_config"] = _runtime_config_for_evaluation(
                {profile_id or case.profile_id for case in cases},
                search_providers=search_providers,
                search_provider_mode=search_provider_mode,
            )
        async with self.service_factory(**service_kwargs) as service:
            if service.model_router is None:
                raise RuntimeError("Evaluation service did not initialize a model router")
            routing_mode = mode or cases[0].mode.value
            generator_routing = service.model_router.describe(routing_mode)
            manifest["generator_routing"] = generator_routing
            manifest["generator_model"] = generator_routing["report_writer"]["model"]
            write_json(manifest_path, manifest)
            rubric_judge = None
            if judge:
                judge_llm = service.model_router.for_tier(judge_tier, role="judge")
                rubric_judge = RubricJudge(
                    judge_llm,
                    batch_timeout_seconds=self.JUDGE_BATCH_TIMEOUT_SECONDS,
                    max_attempts=self.JUDGE_MAX_ATTEMPTS,
                )
                manifest["judge_model"] = judge_llm.model_id
                write_json(manifest_path, manifest)
            campaign_stopped = False
            for case in cases:
                for repetition in range(1, repetitions + 1):
                    key = (case.case_id, repetition)
                    if key in completed:
                        continue
                    if campaign is not None:
                        reservation = campaign.reserve(case.case_id, repetition)
                        if reservation is None:
                            campaign_stopped = True
                            break
                    run_id = _safe_run_id(case.case_id, repetition)
                    started = time.perf_counter()
                    case_deadline = started + self.CASE_TIMEOUT_SECONDS
                    prediction: dict[str, Any] = {
                        "evaluation_id": evaluation_id,
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "run_id": run_id,
                        "mode": mode or case.mode.value,
                        "profile_id": profile_id or case.profile_id,
                    }
                    try:
                        request = case.agent_request(
                            run_id,
                            mode=mode,
                            profile_id=profile_id,
                        )
                        if resume:
                            try:
                                result = await service.resume(run_id)
                            except RunNotFoundError:
                                result = await service.run(request)
                        else:
                            result = await service.run(request)
                        prediction.update(
                            {
                                "status": result.status.value,
                                "citation_gate_passed": result.gate.passed,
                                "artifact_dir": result.artifact_dir,
                                "output_path": result.output_path,
                                "answer_sha256": hashlib.sha256(
                                    result.markdown.encode("utf-8")
                                ).hexdigest(),
                                "usage": result.usage,
                                "error": None,
                            }
                        )
                        if rubric_judge is not None and case.atomic_rubrics:
                            try:
                                remaining_case_seconds = max(
                                    0.001, case_deadline - time.perf_counter()
                                )
                                judged = await asyncio.wait_for(
                                    rubric_judge.judge(case, result.markdown),
                                    timeout=min(
                                        self.JUDGE_TIMEOUT_SECONDS,
                                        remaining_case_seconds,
                                    ),
                                )
                                judged.update(
                                    {
                                        "evaluation_id": evaluation_id,
                                        "repetition": repetition,
                                        "run_id": run_id,
                                        "answer_sha256": hashlib.sha256(
                                            result.markdown.encode("utf-8")
                                        ).hexdigest(),
                                    }
                                )
                                append_unique_jsonl(
                                    judge_results_path,
                                    judged,
                                    key_fields=("case_id", "repetition"),
                                    record_name="judge",
                                )
                                prediction["judge_score"] = judged["overall_score"]
                                prediction["judge_category_scores"] = judged[
                                    "category_scores"
                                ]
                                prediction["judge_error"] = None
                                prediction["judge_usage"] = judged.get("usage", {})
                            except Exception as judge_exc:
                                prediction["judge_score"] = None
                                prediction["judge_error"] = (
                                    f"{type(judge_exc).__name__}: {judge_exc}"
                                )
                    except Exception as exc:
                        prediction.update(
                            {
                                "status": "failed_system",
                                "citation_gate_passed": False,
                                "artifact_dir": None,
                                "output_path": None,
                                "usage": {},
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    prediction["latency_seconds"] = time.perf_counter() - started
                    append_unique_jsonl(
                        predictions_path,
                        prediction,
                        key_fields=("case_id", "repetition"),
                        record_name="prediction",
                    )
                    if campaign is not None:
                        research_usage = prediction.get("usage") or {}
                        judge_usage = prediction.get("judge_usage") or {}
                        consumed_tokens = sum(
                            int(item.get(name, 0) or 0)
                            for item in (research_usage, judge_usage)
                            for name in ("input_tokens", "output_tokens")
                        )
                        campaign.settle(
                            case.case_id,
                            repetition,
                            consumed_tokens=consumed_tokens,
                        )
                        if campaign.data.get("status") == "breached":
                            campaign_stopped = True
                            break
                if campaign_stopped:
                    break

        if campaign is not None:
            campaign.finalize()

        score_evaluation(run_dir)
        self._summarize_judges(run_dir)
        render_report(run_dir)
        return run_dir

    @staticmethod
    def _summarize_judges(run_dir: Path) -> dict[str, Any] | None:
        rows = load_unique_jsonl(
            run_dir / "judge_results.jsonl",
            key_fields=("case_id", "repetition"),
            record_name="judge",
        )
        if not rows:
            return None
        category_totals: dict[str, dict[str, int]] = {}
        for row in rows:
            for category, score in row.get("category_scores", {}).items():
                item = category_totals.setdefault(category, {"passed": 0, "total": 0})
                item["passed"] += int(score.get("passed", 0))
                item["total"] += int(score.get("total", 0))
        passed = sum(int(row.get("passed", 0)) for row in rows)
        total = sum(int(row.get("total", 0)) for row in rows)
        summary = {
            "judged_samples": len(rows),
            "passed": passed,
            "total": total,
            "overall_score": 100.0 * passed / total if total else 0.0,
            "category_scores": {
                category: {
                    **value,
                    "score": (
                        100.0 * value["passed"] / value["total"]
                        if value["total"]
                        else 0.0
                    ),
                }
                for category, value in category_totals.items()
            },
        }
        write_json(run_dir / "judge_summary.json", summary)
        return summary

    async def judge_only(
        self,
        run_dir: str | Path,
        *,
        judge_tier: str | None = None,
    ) -> Path:
        """Judge frozen reports without rerunning generation or retrieval."""

        run_dir = Path(run_dir).resolve()
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Evaluation manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_path = Path(str(manifest.get("dataset_path") or "")).resolve()
        cases = {case.case_id: case for case in load_cases(dataset_path)}
        predictions_path = run_dir / "predictions.jsonl"
        predictions = load_unique_jsonl(
            predictions_path,
            key_fields=("case_id", "repetition"),
            record_name="prediction",
        )
        if not predictions:
            raise ValueError("Evaluation has no frozen predictions to judge")

        tier = judge_tier or str(manifest.get("judge_tier") or "default")
        config = load_runtime_config()
        router = self.model_router_factory(config)
        judge_llm = router.for_tier(tier, role="judge")
        rubric_judge = RubricJudge(
            judge_llm,
            batch_timeout_seconds=self.JUDGE_BATCH_TIMEOUT_SECONDS,
            max_attempts=self.JUDGE_MAX_ATTEMPTS,
        )
        judge_results_path = run_dir / "judge_results.jsonl"
        existing_rows = load_unique_jsonl(
            judge_results_path,
            key_fields=("case_id", "repetition"),
            record_name="judge",
        )
        results_by_key = {
            (str(item.get("case_id")), int(item.get("repetition", 1))): item
            for item in existing_rows
        }

        for prediction in predictions:
            case_id = str(prediction.get("case_id") or "")
            repetition = int(prediction.get("repetition", 1))
            key = (case_id, repetition)
            case = cases.get(case_id)
            if case is None or not case.atomic_rubrics:
                continue
            output_path = Path(str(prediction.get("output_path") or ""))
            if not output_path.is_absolute():
                output_path = (run_dir / output_path).resolve()
            if not output_path.exists() and prediction.get("artifact_dir"):
                output_path = Path(str(prediction["artifact_dir"])) / "report.md"
            try:
                answer = output_path.read_text(encoding="utf-8")
                answer_sha256 = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            except Exception as exc:
                prediction["judge_score"] = None
                prediction["judge_error"] = f"{type(exc).__name__}: {exc}"
                continue
            judged = results_by_key.get(key)
            if judged is None or judged.get("answer_sha256") != answer_sha256:
                try:
                    judged = await asyncio.wait_for(
                        rubric_judge.judge(case, answer),
                        timeout=self.JUDGE_TIMEOUT_SECONDS,
                    )
                    judged.update(
                        {
                            "evaluation_id": manifest.get("evaluation_id"),
                            "repetition": repetition,
                            "run_id": prediction.get("run_id"),
                            "answer_sha256": answer_sha256,
                        }
                    )
                    results_by_key[key] = judged
                except Exception as exc:
                    prediction["judge_score"] = None
                    prediction["judge_error"] = f"{type(exc).__name__}: {exc}"
                    continue
            prediction["judge_score"] = judged["overall_score"]
            prediction["judge_category_scores"] = judged["category_scores"]
            prediction["judge_error"] = None

        ordered_results = [
            results_by_key[key]
            for key in sorted(results_by_key, key=lambda item: (item[0], item[1]))
        ]
        write_jsonl(judge_results_path, ordered_results)
        write_jsonl(predictions_path, predictions)
        manifest["judge_tier"] = tier
        manifest["judge_model"] = judge_llm.model_id
        write_json(manifest_path, manifest)
        score_evaluation(run_dir)
        self._summarize_judges(run_dir)
        render_report(run_dir)
        return run_dir
