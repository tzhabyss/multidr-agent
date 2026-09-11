"""Replay an older Writer against a frozen research checkpoint.

This tool intentionally skips planning, search, fetching, extraction, and
synthesis. It is used to isolate Writer-input regressions from upstream
retrieval changes.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--old-project", required=True, type=Path)
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        choices=("frozen-current", "old-compatible", "no-coverage-noise"),
        default=[],
        help=(
            "Replay variant; repeat to run multiple variants. The historical "
            "two-variant behavior is preserved when omitted."
        ),
    )
    return parser.parse_args()


def _load_pre_writer_state(checkpoint_path: Path) -> dict[str, Any]:
    serializer = JsonPlusSerializer()
    with sqlite3.connect(checkpoint_path) as connection:
        rows = connection.execute(
            "SELECT type, checkpoint FROM checkpoints ORDER BY rowid DESC"
        ).fetchall()
    for value_type, blob in rows:
        checkpoint = serializer.loads_typed((value_type, blob))
        state = checkpoint.get("channel_values", {})
        if state.get("synthesis_decision") and not state.get("structured_output"):
            return state
    raise RuntimeError("No pre-Writer checkpoint with synthesis_decision found")


def _remove_new_coverage_fields(state: dict[str, Any]) -> None:
    decision = state.get("synthesis_decision") or {}
    for item in decision.get("coverage", []):
        if isinstance(item, dict):
            item.pop("required_element_assessments", None)


def _remove_writer_coverage_noise(state: dict[str, Any]) -> None:
    decision = state.get("synthesis_decision") or {}
    decision["coverage"] = []
    state["cell_coverage"] = {}
    state["scope_completeness"] = {}
    for key in ("current_plan",):
        plan = state.get(key)
        if isinstance(plan, dict):
            plan.pop("coverage_blueprint", None)


def _find_local_drb2_dataset(evaluations_root: Path, sample_id: str) -> Path:
    """Find a normalized cached dataset that actually contains the case."""

    dataset_root = evaluations_root / ".datasets"
    preferred = dataset_root / f"drb2-{sample_id}.jsonl"
    candidates = [preferred, *sorted(dataset_root.glob("*.jsonl"))]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if (
                    row.get("case_id") == sample_id
                    and isinstance(row.get("atomic_rubrics"), list)
                ):
                    return candidate
        except (OSError, json.JSONDecodeError):
            continue
    raise FileNotFoundError(
        f"No normalized local DRB2 dataset found for frozen case {sample_id}"
    )
    for plan in state.get("plan_history", []):
        if isinstance(plan, dict):
            plan.pop("coverage_blueprint", None)


class RecordingLLM:
    def __init__(self, target: Any, role: str) -> None:
        self.target = target
        self.role = role
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self.target.model_id

    @property
    def provider_id(self) -> str:
        return str(getattr(self.target, "provider_id", "unknown"))

    async def _record(
        self,
        method: str,
        system: str,
        user: str,
        max_tokens: int,
        invoke: Callable[[], Any],
    ) -> Any:
        started = time.monotonic()
        entry: dict[str, Any] = {
            "role": self.role,
            "method": method,
            "model_id": self.model_id,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
        }
        self.calls.append(entry)
        try:
            result = await invoke()
            response = result[1] if isinstance(result, tuple) else result
            entry.update(
                {
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "input_tokens": getattr(response, "input_tokens", 0),
                    "output_tokens": getattr(response, "output_tokens", 0),
                    "finish_reason": getattr(response, "finish_reason", None),
                }
            )
            return result
        except Exception as exc:
            entry.update(
                {
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int
    ) -> Any:
        return await self._record(
            "complete_json",
            system,
            user,
            max_tokens,
            lambda: self.target.complete_json(system, user, max_tokens=max_tokens),
        )

    async def complete_json_validated(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        validator: Callable[[dict[str, Any]], None],
    ) -> Any:
        return await self._record(
            "complete_json_validated",
            system,
            user,
            max_tokens,
            lambda: self.target.complete_json_validated(
                system,
                user,
                max_tokens=max_tokens,
                validator=validator,
            ),
        )


async def _run_variant(
    *,
    name: str,
    base_state: dict[str, Any],
    old_project: Path,
    source_run_dir: Path,
    output_root: Path,
    remove_new_coverage_fields: bool,
    remove_coverage_noise: bool,
) -> Path:
    from deepresearch.agents import Synthesizer
    from deepresearch.evidence import render_markdown
    from deepresearch.model_routing import ModelRouter
    from deepresearch.runtime_config import load_runtime_config
    from deepresearch.schemas import ClaimEvidenceLink, EvidenceRecord, SourceRecord
    from deepresearch.skill_loader import load_skill_registry

    state = copy.deepcopy(base_state)
    if remove_new_coverage_fields:
        _remove_new_coverage_fields(state)
    if remove_coverage_noise:
        _remove_writer_coverage_noise(state)
    state["run_id"] = f"{state.get('run_id', 'replay')}-{name}"
    state["started_at"] = time.time()

    config = load_runtime_config(old_project / "config" / "runtime.yaml")
    router = ModelRouter(config)
    mode_token = router.activate(str(state.get("request", {}).get("mode") or "standard"))
    writer_llm = RecordingLLM(router.for_stage("report_writer"), "writer")
    reviewer_llm = RecordingLLM(router.for_stage("reviewer"), "reviewer")
    skills = load_skill_registry(
        old_project,
        executable_roles=config.tool_permissions.executable_roles,
        profile_skill_fallbacks={
            profile_id: profile.skill_fallbacks
            for profile_id, profile in config.agent_profiles.items()
            if profile.skill_fallbacks
        },
        profile_skill_provider_modes={
            profile_id: profile.skill_provider_modes
            for profile_id, profile in config.agent_profiles.items()
            if profile.skill_provider_modes
        },
    )
    synthesizer = Synthesizer(
        router.for_stage("synthesizer"),
        skills,
        config,
        writer_llm=writer_llm,
        reviewer_llm=reviewer_llm,
    )

    variant_dir = output_root / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        output, citations, usage, trace = await synthesizer.write(state)
    finally:
        router.reset(mode_token)
    duration = round(time.monotonic() - started, 3)

    links = [ClaimEvidenceLink.model_validate(item) for item in state.get("links", [])]
    evidence = [EvidenceRecord.model_validate(item) for item in state.get("evidence", [])]
    sources = [SourceRecord.model_validate(item) for item in state.get("sources", [])]
    report = render_markdown(output, citations, links, evidence, sources)
    report_path = variant_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    calls = [*writer_llm.calls, *reviewer_llm.calls]
    calls.sort(key=lambda item: (item["role"], item["method"]))
    (variant_dir / "llm_calls.json").write_text(
        json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if writer_llm.calls:
        (variant_dir / "writer_system_prompt.txt").write_text(
            writer_llm.calls[0]["system"], encoding="utf-8"
        )
        (variant_dir / "writer_payload.json").write_text(
            writer_llm.calls[0]["user"], encoding="utf-8"
        )

    metrics = {
        "variant": name,
        "duration_seconds": duration,
        "report_chars": len(report),
        "writer_input_tokens": trace.input_tokens,
        "writer_output_tokens": trace.output_tokens,
        "review_input_tokens": trace.review_input_tokens,
        "review_output_tokens": trace.review_output_tokens,
        "rewrite_input_tokens": trace.rewrite_input_tokens,
        "rewrite_output_tokens": trace.rewrite_output_tokens,
        "recheck_input_tokens": trace.recheck_input_tokens,
        "recheck_output_tokens": trace.recheck_output_tokens,
        "review_applied": trace.review_applied,
        "review_passed": trace.review_passed,
        "review_error": trace.review_error,
        "fallback_used": trace.fallback_used,
        "writer_error": trace.error,
        "total_input_tokens": getattr(usage, "input_tokens", 0),
        "total_output_tokens": getattr(usage, "output_tokens", 0),
    }
    (variant_dir / "replay_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source_manifest = json.loads(
        (source_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    source_prediction = json.loads(
        (source_run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    sample_id = str(source_prediction["case_id"])
    configured_dataset = Path(str(source_manifest.get("dataset_path") or ""))
    if configured_dataset.exists():
        dataset_path = configured_dataset
    else:
        dataset_path = _find_local_drb2_dataset(source_run_dir.parent, sample_id)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "evaluation_id": f"writer-replay-{name}",
            "dataset_path": str(dataset_path.resolve()),
            "judge_model": None,
        }
    )
    prediction = dict(source_prediction)
    prediction.update(
        {
            "evaluation_id": manifest["evaluation_id"],
            "run_id": state["run_id"],
            "output_path": str(report_path.resolve()),
            "judge_score": None,
            "judge_category_scores": None,
            "judge_error": None,
            "latency_seconds": duration,
        }
    )
    (variant_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (variant_dir / "predictions.jsonl").write_text(
        json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (variant_dir / "sample_ids.json").write_text(
        json.dumps([sample_id], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return variant_dir


async def _main() -> None:
    args = _parse_args()
    old_project = args.old_project.resolve()
    sys.path.insert(0, str(old_project))
    state = _load_pre_writer_state(args.checkpoint.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = args.variant or ["old-compatible", "no-coverage-noise"]
    variant_options = {
        "frozen-current": (False, False),
        "old-compatible": (True, False),
        "no-coverage-noise": (True, True),
    }
    results = await asyncio.gather(
        *(
            _run_variant(
                name=name,
                base_state=state,
                old_project=old_project,
                source_run_dir=args.source_run_dir.resolve(),
                output_root=args.output_root.resolve(),
                remove_new_coverage_fields=variant_options[name][0],
                remove_coverage_noise=variant_options[name][1],
            )
            for name in variants
        )
    )
    print("\n".join(str(item) for item in results))


if __name__ == "__main__":
    asyncio.run(_main())
