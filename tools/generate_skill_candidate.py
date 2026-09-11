#!/usr/bin/env python3
"""Explicit offline entrypoint for CoEvoSkills-Lite candidate generation."""

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
    CandidateStore,
    EvolutionEpisodeStore,
)
from deepresearch.skill_loader import load_skill_registry


async def generate(args: argparse.Namespace) -> dict:
    runtime = load_runtime_config(args.config)
    candidate_config = runtime.skill_evolution.candidate_generation.model_copy(
        update={
            "enabled": True,
            "max_total_tokens_per_candidate": args.token_cap,
        }
    )
    store_root = Path(args.store_root)
    if not store_root.is_absolute():
        store_root = ROOT / store_root
    artifact_root = Path(args.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    registry = load_skill_registry(ROOT)
    router = ModelRouter(runtime)
    service = CandidateEvolutionService(
        registry=registry,
        episode_store=EvolutionEpisodeStore(store_root),
        candidate_store=CandidateStore(store_root),
        artifact_root=artifact_root,
        # Candidate calls deliberately bypass Router retries: the phase-two
        # service must see and account for every provider response itself.
        generator_llm=router.resolve_tier(candidate_config.generator_tier),
        verifier_llm=router.resolve_tier(candidate_config.verifier_tier),
        config=candidate_config,
    )
    result = await service.generate_for_episode(
        args.episode_id,
        human_review_approved=args.approve_functional,
    )
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_id")
    parser.add_argument("--config", default=str(ROOT / "config" / "runtime.yaml"))
    parser.add_argument("--store-root", default=".runtime/skill-evolution")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--token-cap", type=int, default=60000)
    parser.add_argument("--approve-functional", action="store_true")
    args = parser.parse_args()
    if args.token_cap <= 0 or args.token_cap > 1_000_000:
        parser.error("--token-cap must be between 1 and 1,000,000")
    asyncio.run(generate(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
