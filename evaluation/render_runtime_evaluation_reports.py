"""Render per-run runtime reports from frozen evaluation artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


RUNS = {
    "remeasure-task75-20260808-v1": {
        "case_id": "task75",
        "label": "Historical low-score remeasurement",
        "history": "Previous score: 15.38% (6/39); this run is directly comparable.",
    },
    "remeasure-task1plus-20260808-v1": {
        "case_id": "task1+",
        "label": "New, previously unmeasured benchmark case",
        "history": "No earlier result exists in the local evaluation history.",
    },
    "remeasure-task44-20260808-v1": {
        "case_id": "task44",
        "label": "Previously measured, non-test-set case",
        "history": "Previous best score: 37.90% (47/124).",
    },
    "remeasure-task62-20260808-v1": {
        "case_id": "task62",
        "label": "Previously measured, non-test-set case",
        "history": "Previous best score: 34.33% (23/67).",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_event_lines(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A malformed diagnostic line must not discard the completed run's
            # valid event records.
            continue
    return events


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def elapsed(start: str | None, end: str | None) -> str:
    if not start or not end:
        return "not separately recorded"
    seconds = (parse_time(end) - parse_time(start)).total_seconds()
    return f"{seconds:.3f} s ({seconds / 60:.2f} min)"


def event_first(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("event_type") == event_type), None)


def event_all(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def trace_counts(sample: Path) -> dict[str, Any]:
    traces = sample / "details" / "traces"
    fetches = load_json(traces / "fetch.json")
    extractions = load_json(traces / "extraction.json")
    searches = load_json(traces / "search.json")
    workers = load_json(traces / "worker.json")
    worker_phases = Counter(item.get("phase") or "unknown" for item in workers)
    fetch_statuses = Counter(item.get("status") or "unknown" for item in fetches)
    extraction_errors = sum(bool(item.get("error")) for item in extractions)
    accepted = sum(int(item.get("accepted_count") or 0) for item in extractions)
    return {
        "search_trace_entries": len(searches),
        "fetch_trace_entries": len(fetches),
        "fetch_statuses": fetch_statuses,
        "extraction_trace_entries": len(extractions),
        "accepted_claims": accepted,
        "extraction_errors": extraction_errors,
        "worker_phases": worker_phases,
    }


def render(run_root: Path, metadata: dict[str, str]) -> Path:
    summary = load_json(run_root / "summary.json")
    judge = load_json(run_root / "judge_summary.json")
    manifest = load_json(run_root / "manifest.json")
    sample = next((run_root / "samples").iterdir())
    sample_manifest = load_json(sample / "details" / "manifest.json")
    request = load_json(sample / "details" / "input" / "request.json")
    events = load_event_lines(sample / "details" / "runtime" / "events.jsonl")
    usage = load_json(sample / "details" / "runtime" / "usage.json")
    report_trace = load_json(sample / "details" / "traces" / "report.json")
    final_report = (sample / "report.md").read_text(encoding="utf-8")
    counts = trace_counts(sample)

    started = event_first(events, "run_started")
    terminal = event_first(events, "run_terminal")
    plans = event_all(events, "plan_created")
    syntheses = event_all(events, "synthesis_completed")
    report_event = event_first(events, "report_synthesized")
    citation_event = event_first(events, "citations_attached")
    waves = event_all(events, "wave_merged")
    workers = event_all(events, "worker_completed")

    usage_totals: dict[str, int] = defaultdict(int)
    for item in usage:
        category = item.get("category") or "unknown"
        usage_totals[f"{category}_input"] += int(item.get("input_tokens") or 0)
        usage_totals[f"{category}_output"] += int(item.get("output_tokens") or 0)
        usage_totals[f"{category}_searches"] += int(item.get("searches") or 0)
        usage_totals[f"{category}_fetches"] += int(item.get("fetches") or 0)

    first_plan = plans[0] if plans else None
    last_synthesis = syntheses[-1] if syntheses else None
    lines = [
        f"# Runtime Evaluation Report: {run_root.name}",
        "",
        f"- **Purpose:** {metadata['label']}",
        f"- **Comparison:** {metadata['history']}",
        f"- **Case:** `{metadata['case_id']}`",
        f"- **Run ID:** `{sample_manifest.get('run_id', 'unknown')}`",
        f"- **Mode/profile:** `{sample_manifest.get('mode')}` / `{sample_manifest.get('profile_id')}`",
        f"- **Terminal status:** `{sample_manifest.get('status')}`",
        f"- **Artifact directory:** `{sample}`",
        "",
        "## Score",
        "",
        f"- Overall Judge score: **{judge['overall_score']:.2f}%** ({judge['passed']}/{judge['total']} rubric items).",
    ]
    for category, value in judge["category_scores"].items():
        lines.append(f"- {category}: {value['score']:.2f}% ({value['passed']}/{value['total']}).")
    lines += [
        f"- Deterministic status: verified {summary['verified_run_rate']:.0%}, partial {summary['partial_run_rate']:.0%}, failed {summary['failure_rate']:.0%}.",
        f"- Citation grounding precision: {summary['citation_grounding_precision']:.2%}; unsupported-claim rate: {summary['unsupported_claim_rate']:.2%}.",
        "",
        "## Timing",
        "",
        "All timestamps are ISO 8601 in `+08:00`. Durations are calculated from the artifact event times.",
        "",
        "| Stage | Start | End | Duration | Notes |",
        "|---|---|---|---|---|",
        f"| Initial planning | {started.get('timestamp') if started else 'n/a'} | {first_plan.get('timestamp') if first_plan else 'n/a'} | {elapsed(started.get('timestamp') if started else None, first_plan.get('timestamp') if first_plan else None)} | First accepted planner output |",
    ]
    for index, plan in enumerate(plans, 1):
        synthesis = next((item for item in syntheses if item.get('payload', {}).get('round_no') == index), None)
        lines.append(
            f"| Research round {index} | {plan.get('timestamp')} | {synthesis.get('timestamp') if synthesis else 'n/a'} | {elapsed(plan.get('timestamp'), synthesis.get('timestamp') if synthesis else None)} | {plan.get('payload', {}).get('task_count', 0)} planned tasks; synthesis decision `{synthesis.get('payload', {}).get('decision') if synthesis else 'n/a'}` |"
        )
    lines += [
        f"| Writer + reviewer (combined) | {last_synthesis.get('timestamp') if last_synthesis else 'n/a'} | {report_event.get('timestamp') if report_event else 'n/a'} | {elapsed(last_synthesis.get('timestamp') if last_synthesis else None, report_event.get('timestamp') if report_event else None)} | No separate writer/reviewer start/end events were emitted; `review_applied={report_trace.get('review_applied')}` |",
        f"| Citation attachment + finalization | {report_event.get('timestamp') if report_event else 'n/a'} | {terminal.get('timestamp') if terminal else 'n/a'} | {elapsed(report_event.get('timestamp') if report_event else None, terminal.get('timestamp') if terminal else None)} | citation event at `{citation_event.get('timestamp') if citation_event else 'n/a'}` |",
        f"| End-to-end (artifact events) | {started.get('timestamp') if started else 'n/a'} | {terminal.get('timestamp') if terminal else 'n/a'} | {elapsed(started.get('timestamp') if started else None, terminal.get('timestamp') if terminal else None)} | Evaluation summary p50 latency: {summary['p50_latency_seconds']:.3f} s |",
        "",
        "## DAG And Worker Waves",
        "",
        f"- Planner rounds: {len(plans)}; worker-completed events: {len(workers)}; merged waves: {len(waves)}.",
        f"- Worker completion sufficiency: {sum(bool(item.get('payload', {}).get('sufficiency')) for item in workers)}/{len(workers)} marked sufficient.",
        "",
        "| Wave | Scheduled | Completed | Duration | Tasks | Sources | Evidence | Claims | Requirement support |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    schedule_by_wave = {
        item.get("payload", {}).get("wave_id"): item for item in event_all(events, "wave_scheduled")
    }
    for wave in waves:
        payload = wave.get("payload", {})
        wave_id = payload.get("wave_id")
        scheduled = schedule_by_wave.get(wave_id, {}).get("timestamp", "n/a")
        lines.append(
            "| {wave} | {start} | {end} | {duration:.3f} s | {tasks} | {sources} | {evidence} | {claims} | {support} |".format(
                wave=wave_id,
                start=scheduled,
                end=wave.get("timestamp"),
                duration=float(payload.get("elapsed_seconds") or 0),
                tasks=", ".join(payload.get("task_ids") or []),
                sources=payload.get("source_delta", 0),
                evidence=payload.get("evidence_delta", 0),
                claims=payload.get("claim_delta", 0),
                support=payload.get("requirement_support_delta", 0),
            )
        )
    lines += [
        "",
        "## Search, Fetch, Extraction And Token Use",
        "",
        f"- Usage ledger: {usage_totals['llm_input'] + usage_totals['llm_output']:,} tokens ({usage_totals['llm_input']:,} input, {usage_totals['llm_output']:,} output), {usage_totals['search_searches']} searches, {usage_totals['fetch_fetches']} fetches.",
        f"- Trace records: {counts['search_trace_entries']} search batches, {counts['fetch_trace_entries']} fetch attempts, {counts['extraction_trace_entries']} extraction chunks.",
        f"- Fetch statuses: {', '.join(f'{key}={value}' for key, value in sorted(counts['fetch_statuses'].items())) or 'none'}.",
        f"- Extraction: {counts['accepted_claims']} accepted claims across chunks; {counts['extraction_errors']} extraction chunks with an explicit error.",
        f"- Worker LLM trace phases: {', '.join(f'{key}={value}' for key, value in sorted(counts['worker_phases'].items()))}.",
        "",
        "## Runtime Configuration",
        "",
        f"- Generator routing: planner `{manifest['generator_routing']['planner']['model']}`, worker `{manifest['generator_routing']['worker_agent_loop']['model']}`, writer `{manifest['generator_routing']['report_writer']['model']}`.",
        f"- Judge: `{manifest['judge_model']}` at `{manifest['judge_tier']}` tier.",
        f"- Research limits: {sample_manifest['budget']['max_wall_time_seconds']} s wall time, {sample_manifest['budget']['max_total_searches']} searches, {sample_manifest['budget']['max_total_fetches']} fetches.",
        f"- Writer model: `{report_trace.get('model_id')}`; review model: `{report_trace.get('review_model_id')}`; fallback used: `{report_trace.get('fallback_used')}`; review error: `{report_trace.get('review_error')}`.",
        "",
        "## Final Generated Report",
        "",
        final_report.rstrip(),
        "",
    ]
    destination = run_root / "runtime-evaluation-report.md"
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return destination


def main() -> None:
    root = Path("artifacts/evaluations")
    for run, metadata in RUNS.items():
        print(render(root / run, metadata))


if __name__ == "__main__":
    main()
