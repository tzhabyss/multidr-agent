#!/usr/bin/env python3
"""Explicit offline CLI for replay-backed CoEvoSkills verifier evolution."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.model_routing import ModelRouter
from deepresearch.runtime_config import load_runtime_config
from deepresearch.skill_evolution import (
    CandidateEvolutionService,
    CandidateReplayObservation,
    CandidateStore,
    EvolutionEpisodeStore,
    VerifierCoEvolutionService,
    VerifierEvolutionStore,
)
from deepresearch.skill_loader import load_skill_registry


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _service(args: argparse.Namespace) -> VerifierCoEvolutionService:
    runtime = load_runtime_config(args.config)
    candidate_config = runtime.skill_evolution.candidate_generation.model_copy(
        update={"enabled": True}
    )
    coevolution_config = runtime.skill_evolution.verifier_coevolution.model_copy(
        update={
            "enabled": True,
            "max_total_tokens_per_gap": args.token_cap,
        }
    )
    router = ModelRouter(runtime)
    root = _resolve(args.store_root)
    episode_store = EvolutionEpisodeStore(root)
    candidate_store = CandidateStore(root)
    candidate_service = CandidateEvolutionService(
        registry=load_skill_registry(ROOT),
        episode_store=episode_store,
        candidate_store=candidate_store,
        artifact_root=_resolve(args.artifact_root),
        generator_llm=router.resolve_tier(candidate_config.generator_tier),
        verifier_llm=router.resolve_tier(candidate_config.verifier_tier),
        config=candidate_config,
    )
    return VerifierCoEvolutionService(
        candidate_service=candidate_service,
        candidate_store=candidate_store,
        episode_store=episode_store,
        verifier_store=VerifierEvolutionStore(root),
        generator_llm=router.resolve_tier(coevolution_config.generator_tier),
        config=coevolution_config,
    )


async def run(args: argparse.Namespace) -> dict:
    service = _service(args)
    if args.action == "evaluate":
        observation = CandidateReplayObservation.model_validate_json(
            Path(args.observation).read_text(encoding="utf-8")
        )
        result = service.evaluate_replay(observation)
    elif args.action == "propose-verifier":
        result = await service.generate_verifier_suite(args.gap_id)
    elif args.action == "activate-verifier":
        result = service.activate_verifier_suite(
            args.suite_id,
            args.version,
            human_review_approved=args.approve,
        )
    else:
        result = await service.revise_candidate_from_gap(args.gap_id)
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "runtime.yaml"))
    parser.add_argument("--store-root", default=".runtime/skill-evolution")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--token-cap", type=int, default=30000)
    subparsers = parser.add_subparsers(dest="action", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("observation")

    propose = subparsers.add_parser("propose-verifier")
    propose.add_argument("gap_id")

    activate = subparsers.add_parser("activate-verifier")
    activate.add_argument("suite_id")
    activate.add_argument("version")
    activate.add_argument("--approve", action="store_true")

    revise = subparsers.add_parser("revise-candidate")
    revise.add_argument("gap_id")

    args = parser.parse_args()
    if args.token_cap <= 0 or args.token_cap > 100000:
        parser.error("--token-cap must be between 1 and 100,000")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

