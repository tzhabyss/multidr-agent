from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.runtime_config import VerifierCoEvolutionRuntimeConfig
from deepresearch.skill_evolution import (
    CandidateMaterializationError,
    CandidateRecord,
    CandidateSkillMaterializer,
    CandidateStore,
    EvolutionEpisode,
    EvolutionEpisodeStore,
    LocalMicroFixture,
    LocalReplayArmResult,
    LocalSkillReplayRunner,
    VerifierCoEvolutionService,
    VerifierEvolutionStore,
)
from deepresearch.skill_loader import load_skill_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UnusedLLM:
    model_id = "unused"

    async def complete_json(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("local deterministic replay must not call an LLM")


def _episode() -> EvolutionEpisode:
    return EvolutionEpisode(
        episode_id="episode-duplicate-query",
        run_id="run-duplicate-query",
        task_id="task-search",
        target_skill_id="web-search",
        target_skill_version="1.0.0",
        runtime_layer="access",
        capability_granularity="atomic",
        candidate_policy="automatic_candidate_allowed",
        trigger_signal_ids=["signal-duplicate-query"],
        trigger_types=["duplicate_query"],
        local_context={"duplicate_query_ratio": 0.5},
        local_result={"duplicate_queries": 1},
        confidence=0.9,
    )


def _candidate(*, requested_permissions: list[str] | None = None) -> CandidateRecord:
    return CandidateRecord.model_validate(
        {
            "candidate_id": "candidate-duplicate-query",
            "source_episode_id": "episode-duplicate-query",
            "run_id": "run-duplicate-query",
            "target_skill_id": "web-search",
            "parent_skill_version": "1.0.0",
            "capability_granularity": "atomic",
            "status": "accept_for_replay",
            "final_patch": {
                "candidate_id": "candidate-duplicate-query",
                "target_skill_id": "web-search",
                "parent_skill_version": "1.0.0",
                "capability_granularity": "atomic",
                "trigger_conditions": ["exact duplicate query in one task"],
                "applicability_preconditions": ["query constraints are unchanged"],
                "operations": [
                    {
                        "operation_id": "op-deduplicate",
                        "kind": "add",
                        "target_section": "Web Search",
                        "anchor": "# Web Search",
                        "new_text": (
                            "Before executing a query, reuse an identical prior query "
                            "when entity, field, date, locale, and source constraints are unchanged."
                        ),
                        "rationale": "avoid duplicate provider calls",
                    }
                ],
                "expected_improvements": ["reduce duplicate queries"],
                "risks": ["different constraints must remain distinct"],
                "source_episode_ids": ["episode-duplicate-query"],
                "requested_permissions": requested_permissions or [],
                "patch_fingerprint": "fixture-patch-fingerprint",
            },
        }
    )


def _materializer(tmp_path: Path) -> CandidateSkillMaterializer:
    return CandidateSkillMaterializer(
        registry=load_skill_registry(PROJECT_ROOT),
        replay_root=tmp_path / "local-replays",
    )


def test_candidate_materialization_is_isolated_and_loadable(tmp_path: Path):
    source = PROJECT_ROOT / "skills" / "web-search" / "SKILL.md"
    source_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()

    result = _materializer(tmp_path).materialize(
        _candidate(), replay_id="replay-isolated"
    )

    assert result.stable_sha256 != result.candidate_sha256
    assert "reuse an identical prior query" not in result.stable_skill_path.read_text()
    assert "reuse an identical prior query" in result.candidate_skill_path.read_text()
    assert __import__("hashlib").sha256(source.read_bytes()).hexdigest() == source_hash
    candidate_registry = load_skill_registry(result.candidate_project_root)
    assert candidate_registry.get("web-search") is not None
    assert candidate_registry.get("web-search").skill_version.startswith(
        "1.0.0+candidate."
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stable_skill_mutated"] is False
    assert manifest["production_loading_enabled"] is False

    resumed = _materializer(tmp_path).materialize(
        _candidate(), replay_id="replay-isolated"
    )
    assert resumed.manifest_path == result.manifest_path
    assert resumed.stable_sha256 == result.stable_sha256
    assert resumed.candidate_sha256 == result.candidate_sha256


def test_candidate_materialization_rejects_incomplete_existing_workspace(
    tmp_path: Path,
):
    incomplete = tmp_path / "local-replays" / "replay-incomplete"
    incomplete.mkdir(parents=True)

    with pytest.raises(CandidateMaterializationError, match="incomplete"):
        _materializer(tmp_path).materialize(
            _candidate(), replay_id="replay-incomplete"
        )


def test_candidate_materialization_rejects_permission_expansion(tmp_path: Path):
    with pytest.raises(CandidateMaterializationError, match="permissions"):
        _materializer(tmp_path).materialize(
            _candidate(requested_permissions=["unrestricted_network"]),
            replay_id="replay-forbidden",
        )


@pytest.mark.asyncio
async def test_local_replay_reaches_project_oracle_and_is_idempotent(tmp_path: Path):
    root = tmp_path / "skill-evolution"
    episode = _episode()
    candidate = _candidate()
    episode_store = EvolutionEpisodeStore(root)
    candidate_store = CandidateStore(root)
    episode_store.write_run(episode.run_id, [episode])
    candidate_store.write(candidate)
    verifier_store = VerifierEvolutionStore(root)
    verifier_service = VerifierCoEvolutionService(
        candidate_service=None,  # revision is outside this deterministic replay
        candidate_store=candidate_store,
        episode_store=episode_store,
        verifier_store=verifier_store,
        generator_llm=UnusedLLM(),
        config=VerifierCoEvolutionRuntimeConfig(enabled=True),
    )

    async def executor(fixture, instructions: str, arm: str):
        improved = "reuse an identical prior query" in instructions
        return LocalReplayArmResult(
            arm=arm,
            metrics={"duplicate_queries": 0.0 if improved else 1.0},
            hard_checks={"unique_queries_preserved": True},
            output={"executed_queries": 2 if improved else 3},
        )

    runner = LocalSkillReplayRunner(
        materializer=_materializer(tmp_path),
        verifier_service=verifier_service,
        executor=executor,
    )
    fixture = LocalMicroFixture(
        fixture_id="MICRO-01-exact-duplicate-v1",
        target_skill_id="web-search",
        trigger_type="duplicate_query",
        input_snapshot={
            "queries": ["SkillX benchmark", "SkillX benchmark", "SkillX GitHub"]
        },
        target_metric="duplicate_queries",
        invariants=["unique_queries_preserved"],
    )

    first = await runner.run(
        fixture=fixture, candidate=candidate, episode=episode
    )
    second = await runner.run(
        fixture=fixture, candidate=candidate, episode=episode
    )

    assert first.result.status == "ready_for_shadow"
    assert first.result.oracle_result.decision == "pass"
    assert first.result.observation.stable_metrics["duplicate_queries"] == 1.0
    assert first.result.observation.candidate_metrics["duplicate_queries"] == 0.0
    assert Path(first.observation_path).is_file()
    assert second.result.reused_existing is True
    assert verifier_store.read_replay(first.replay_id) is not None


@pytest.mark.asyncio
async def test_local_replay_creates_verifier_gap_on_invariant_regression(tmp_path: Path):
    root = tmp_path / "skill-evolution"
    episode = _episode()
    candidate = _candidate()
    episode_store = EvolutionEpisodeStore(root)
    candidate_store = CandidateStore(root)
    episode_store.write_run(episode.run_id, [episode])
    candidate_store.write(candidate)
    verifier_store = VerifierEvolutionStore(root)
    service = VerifierCoEvolutionService(
        candidate_service=None,
        candidate_store=candidate_store,
        episode_store=episode_store,
        verifier_store=verifier_store,
        generator_llm=UnusedLLM(),
        config=VerifierCoEvolutionRuntimeConfig(enabled=True),
    )

    async def executor(fixture, instructions: str, arm: str):
        return LocalReplayArmResult(
            arm=arm,
            metrics={"duplicate_queries": 0.0 if arm == "candidate" else 1.0},
            hard_checks={"unique_queries_preserved": arm == "stable"},
        )

    result = await LocalSkillReplayRunner(
        materializer=_materializer(tmp_path),
        verifier_service=service,
        executor=executor,
    ).run(
        fixture=LocalMicroFixture(
            fixture_id="MICRO-01-regression-v1",
            target_skill_id="web-search",
            trigger_type="duplicate_query",
            input_snapshot={"queries": ["results 2025", "results 2026"]},
            target_metric="duplicate_queries",
            invariants=["unique_queries_preserved"],
        ),
        candidate=candidate,
        episode=episode,
    )

    assert result.result.status == "hold"
    assert result.result.verifier_gap_id
    assert verifier_store.read_gap(result.result.verifier_gap_id) is not None
