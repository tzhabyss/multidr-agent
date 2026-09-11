"""Deterministic Markdown report renderer; no LLM rewrites metric values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(run_dir: str | Path, output: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    judge_summary_path = run_dir / "judge_summary.json"
    judge_summary = (
        json.loads(judge_summary_path.read_text(encoding="utf-8"))
        if judge_summary_path.exists()
        else None
    )
    campaign_path = run_dir / "campaign_token_ledger.json"
    campaign = (
        json.loads(campaign_path.read_text(encoding="utf-8"))
        if campaign_path.exists()
        else None
    )
    output = Path(output) if output else run_dir / "evaluation_report.md"
    metrics = [
        ("Verified Run Rate", "verified_run_rate"),
        ("Partial Run Rate", "partial_run_rate"),
        ("Failure Rate", "failure_rate"),
        ("Citation Grounding Precision", "citation_grounding_precision"),
        ("Quote Match Rate", "quote_match_rate"),
        ("Unsupported Claim Rate", "unsupported_claim_rate"),
        ("High-impact Claim Coverage", "high_impact_claim_coverage"),
        ("Provenance Completeness", "provenance_completeness"),
        ("Budget Compliance Rate", "budget_compliance_rate"),
        ("Terminal Event Exactly Once", "terminal_event_exactly_once_rate"),
        ("Mean Tokens", "mean_total_tokens"),
        ("Mean Searches", "mean_searches"),
        ("Mean Fetches", "mean_fetches"),
        ("P50 Latency (s)", "p50_latency_seconds"),
        ("P95 Latency (s)", "p95_latency_seconds"),
    ]
    lines = [
        "# Deep Research V1 Evaluation Report",
        "",
        "## Protocol",
        "",
        f"- Evaluation ID: `{manifest['evaluation_id']}`",
        f"- Dataset: `{manifest['dataset_path']}`",
        f"- Dataset SHA256: `{manifest['dataset_sha256']}`",
        f"- Dataset revision: `{manifest.get('dataset_revision') or 'unspecified'}`",
        f"- Track: `{manifest['track']}`",
        f"- Variant: `{manifest['variant']}`",
        f"- Repetitions: `{manifest['repetitions']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Generator model: `{manifest['generator_model']}`",
        f"- Generator routing: `{json.dumps(manifest.get('generator_routing'), ensure_ascii=False)}`",
        "",
        "## Deterministic Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {_format(summary.get(key))} |" for label, key in metrics)
    if campaign is not None:
        lines.extend(
            [
                "",
                "## Campaign Token Ledger",
                "",
                f"- Status: `{campaign.get('status')}`",
                f"- Budget: `{campaign.get('max_total_tokens', 0)}`",
                f"- Consumed: `{campaign.get('consumed_tokens', 0)}`",
                f"- Reserved: `{campaign.get('reserved_tokens', 0)}`",
                f"- Remaining: `{campaign.get('remaining_tokens', 0)}`",
                f"- Stop reason: `{campaign.get('stop_reason') or 'none'}`",
            ]
        )
    if judge_summary is not None:
        lines.extend(
            [
                "",
                "## Rubric Judge",
                "",
                f"- Judge model: `{manifest.get('judge_model') or 'unknown'}`",
                f"- Overall score: `{_format(judge_summary.get('overall_score'))}`",
                f"- Rubrics passed: `{judge_summary.get('passed', 0)}/{judge_summary.get('total', 0)}`",
                "",
                "| Category | Passed | Total | Score |",
                "|---|---:|---:|---:|",
            ]
        )
        for category, score in judge_summary.get("category_scores", {}).items():
            lines.append(
                f"| {category} | {score.get('passed', 0)} | {score.get('total', 0)} | "
                f"{_format(score.get('score'))} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These values are computed from this project's typed runtime ledger and artifacts. "
            "They are not official DeepResearch Bench FACT, RACE, BrowseComp-Plus, or FRAMES scores.",
            "",
            "The deterministic gate verifies citation resolution, exact quote provenance, and numeric "
            "grounding. It does not by itself prove full semantic factual correctness.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
