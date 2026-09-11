#!/usr/bin/env python3
"""Run a compact real-model production Reviewer completeness breakpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.agent_response_contracts import validate_review_payload
from deepresearch.model_routing import ModelRouter
from deepresearch.prompts import REPORT_REVIEW_SYSTEM_PROMPT
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import WriterReview


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


async def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture_path = _resolve(args.fixture)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload = dict(fixture["payload"])
    payload["output_schema"] = {
        "decision": "pass or repair",
        "issues": [
            {
                "location": "section, paragraph, table, row, or cell",
                "severity": "low, medium, or high",
                "issue_type": (
                    "grounding, completeness, structure, accuracy, "
                    "analysis_depth, reasoning_gap, or overclaim"
                ),
                "message": "concise explanation",
            }
        ],
        "revised_draft": None,
    }
    user = json.dumps(payload, ensure_ascii=False)
    runtime = load_runtime_config(_resolve(args.config))
    llm = ModelRouter(runtime).for_stage("reviewer")
    oracle = fixture["oracle"]
    rows: list[dict[str, Any]] = []
    total_tokens = 0
    for attempt in range(1, int(fixture.get("attempts", 3)) + 1):
        reservation = len(REPORT_REVIEW_SYSTEM_PROMPT) + len(user) + args.max_tokens
        if total_tokens + reservation > args.token_cap:
            raise RuntimeError("Reviewer microcase token preflight exceeded cap")
        value, response = await llm.complete_json_validated(
            REPORT_REVIEW_SYSTEM_PROMPT,
            user,
            max_tokens=args.max_tokens,
            validator=validate_review_payload,
        )
        review = WriterReview.model_validate(value)
        total_tokens += response.input_tokens + response.output_tokens
        issue_types = {str(item.issue_type).casefold() for item in review.issues}
        issue_text = " ".join(item.message for item in review.issues).casefold()
        checks = {
            "repair_requested": review.decision == oracle["expected_decision"],
            "accepted_issue_type": bool(
                issue_types
                & {str(item).casefold() for item in oracle["accepted_issue_types"]}
            ),
            "missing_requirement_identified": any(
                str(term).casefold() in issue_text
                for term in oracle["message_must_contain_any"]
            ),
        }
        rows.append(
            {
                "attempt": attempt,
                "decision": review.decision,
                "issues": [item.model_dump(mode="json") for item in review.issues],
                "checks": checks,
                "passed": all(checks.values()),
                "tokens": response.input_tokens + response.output_tokens,
            }
        )
    result = {
        "schema_version": "1.0",
        "case_id": fixture["case_id"],
        "model_id": llm.model_id,
        "attempts": rows,
        "pass_count": sum(item["passed"] for item in rows),
        "attempt_count": len(rows),
        "stable_failure_reproduced": any(not item["passed"] for item in rows),
        "token_usage": total_tokens,
        "token_cap": args.token_cap,
        "candidate_generation_enabled": False,
        "stable_mutation_enabled": False,
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        default="evaluation/fixtures/reviewer/task40-contradictory-statements-gap-20260906-v1.json",
    )
    parser.add_argument("--config", default="config/runtime.yaml")
    parser.add_argument(
        "--output",
        default="output/skill-micro-eval/task40-reviewer-gap-20260906/result.json",
    )
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--token-cap", type=int, default=50000)
    args = parser.parse_args()
    if not 1 <= args.token_cap <= 1_000_000:
        parser.error("--token-cap must be between 1 and 1,000,000")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
