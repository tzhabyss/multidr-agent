"""Evaluation command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import sys
from pathlib import Path
from typing import Sequence

from .io import load_cases
from .metrics import score_evaluation
from .report import render_report
from .runner import EvaluationRunner
from .drb2 import prepare_drb2_dataset
from .replay import build_candidate_replay_observation
from deepresearch.skill_evolution.models import CandidateRecord


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepresearch-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate a JSONL dataset")
    doctor.add_argument("--dataset", required=True, type=Path)

    run = subparsers.add_parser("run", help="run a JSONL evaluation")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--benchmark", choices=("drb2",))
    run.add_argument("--output-root", type=Path, default=Path("artifacts/evaluations"))
    run.add_argument("--evaluation-id")
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--track", default="general_research")
    run.add_argument("--variant", default="p1_full")
    run.add_argument("--dataset-revision")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--case-id", action="append", default=[])
    run.add_argument("--limit", type=int)
    run.add_argument("--mode", choices=("quick", "standard", "deep", "max"))
    run.add_argument("--profile")
    run.add_argument("--judge", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--judge-tier", choices=("lite", "default", "craft"), default="default")
    run.add_argument(
        "--search-provider",
        action="append",
        default=[],
        help="profile search child skill id; repeat to select multiple providers",
    )
    run.add_argument(
        "--search-provider-mode",
        choices=("fallback", "fanout"),
        default="fanout",
        help="try providers in order or query all selected providers",
    )
    run.add_argument(
        "--campaign-token-budget",
        type=int,
        help="maximum provider-reported tokens admitted across the evaluation",
    )
    run.add_argument(
        "--max-tokens-per-sample",
        type=int,
        help="reservation required before starting each sample",
    )

    score = subparsers.add_parser("score", help="recompute deterministic scores")
    score.add_argument("--run-dir", required=True, type=Path)

    report = subparsers.add_parser("report", help="render the Markdown report")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--output", type=Path)

    judge_only = subparsers.add_parser(
        "judge-only", help="judge existing frozen predictions without generation"
    )
    judge_only.add_argument("--run-dir", required=True, type=Path)
    judge_only.add_argument(
        "--judge-tier", choices=("lite", "default", "craft")
    )
    replay = subparsers.add_parser(
        "replay-observation",
        help="build a CoEvoSkills observation from paired frozen evaluations",
    )
    replay.add_argument("--stable-run-dir", required=True, type=Path)
    replay.add_argument("--candidate-run-dir", required=True, type=Path)
    replay.add_argument("--candidate-record", required=True, type=Path)
    replay.add_argument("--case-id", required=True)
    replay.add_argument("--repetition", type=int, default=1)
    replay.add_argument("--output", required=True, type=Path)
    return parser


async def _run(args: argparse.Namespace) -> int:
    dataset = args.dataset
    track = args.track
    profile = args.profile
    judge = args.judge
    if args.benchmark == "drb2":
        selection = json.dumps(
            {"case_ids": args.case_id, "limit": args.limit}, sort_keys=True
        ).encode("utf-8")
        label = hashlib.sha256(selection).hexdigest()[:10]
        prepared_dataset = args.output_root / ".datasets" / f"drb2-{label}.jsonl"
        # The selection hash fully identifies the normalized benchmark slice.
        # Reuse it for fresh evaluation IDs as well as resumed runs; requiring
        # a network download for every new run makes offline/repeated scoring
        # fail even when the exact immutable input is already cached.
        if prepared_dataset.exists() and prepared_dataset.stat().st_size:
            dataset = prepared_dataset
        else:
            dataset = await prepare_drb2_dataset(
                prepared_dataset,
                case_ids=args.case_id,
                limit=args.limit,
            )
        track = "drb2"
        profile = profile or "drb_evaluation"
        judge = True if judge is None else judge
    elif args.case_id or args.limit:
        raise ValueError("--case-id and --limit require --benchmark")
    judge = False if judge is None else judge
    run_dir = await EvaluationRunner().run(
        dataset_path=dataset,
        output_root=args.output_root,
        evaluation_id=args.evaluation_id,
        repetitions=args.repetitions,
        seed=args.seed,
        resume=args.resume,
        track=track,
        variant=args.variant,
        dataset_revision=args.dataset_revision,
        mode=args.mode,
        profile_id=profile,
        judge=judge,
        judge_tier=args.judge_tier,
        search_providers=args.search_provider or None,
        search_provider_mode=args.search_provider_mode,
        campaign_token_budget=getattr(args, "campaign_token_budget", None),
        max_tokens_per_sample=getattr(args, "max_tokens_per_sample", None),
    )
    print(run_dir)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            cases = load_cases(args.dataset)
            print(f"valid cases: {len(cases)}")
            return 0
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "score":
            print(json.dumps(score_evaluation(args.run_dir), ensure_ascii=False, indent=2))
            return 0
        if args.command == "judge-only":
            output = asyncio.run(
                EvaluationRunner().judge_only(
                    args.run_dir,
                    judge_tier=args.judge_tier,
                )
            )
            print(output)
            return 0
        if args.command == "replay-observation":
            candidate = CandidateRecord.model_validate_json(
                args.candidate_record.read_text(encoding="utf-8")
            )
            observation = build_candidate_replay_observation(
                stable_run_dir=args.stable_run_dir,
                candidate_run_dir=args.candidate_run_dir,
                candidate=candidate,
                case_id=args.case_id,
                repetition=args.repetition,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                observation.model_dump_json(indent=2), encoding="utf-8"
            )
            print(args.output)
            return 0
        output = render_report(args.run_dir, args.output)
        print(output)
        return 0
    except (OSError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
