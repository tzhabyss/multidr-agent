from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.runtime_config import load_runtime_config
from deepresearch.skill_evolution import (
    CandidateEvolutionService,
    CandidateReplayObservation,
    CandidateStore,
    EvolutionEpisode,
    EvolutionEpisodeStore,
    VerifierCoEvolutionDisabled,
    VerifierCoEvolutionService,
    VerifierEvolutionStore,
)
from deepresearch.skill_loader import load_skill_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLM:
    def __init__(self, model_id: str, responses: list[dict]):
        self.model_id = model_id
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete_json(self, system: str, user: str, *, max_tokens: int):
        self.calls.append(json.loads(user))
        value = self.responses.pop(0)
        return value, LLMResponse(
            json.dumps(value), input_tokens=100, output_tokens=50
        )

    async def complete_text(self, system: str, user: str, *, max_tokens: int):
        raise AssertionError("co-evolution must use structured JSON")

    async def complete_with_tools(self, *, messages, tools, max_tokens: int):
        raise AssertionError("co-evolution cannot call tools")


def _episode() -> EvolutionEpisode:
    return EvolutionEpisode(
        episode_id="episode-coevo",
        run_id="run-coevo",
        task_id="task-1",
        target_skill_id="safe-fetch",
        target_skill_version="1.0.0",
        runtime_layer="access",
        capability_granularity="atomic",
        candidate_policy="automatic_candidate_allowed",
        trigger_signal_ids=["signal-1"],
        trigger_types=["fetch_route_failure"],
        scope={"type": "url", "value": "https://example.test/paper"},
        local_context={"summary": "A recoverable static fetch route failed."},
        actions=[{"url": "https://example.test/paper"}],
        local_result={"observed_failure_or_low_gain": True},
        evidence_refs=[
            {
                "artifact_path": "details/traces/fetch.json",
                "record_type": "fetch_trace",
                "record_id": "fetch-1",
            }
        ],
        replay_spec={
            "artifact_paths": ["details/traces/fetch.json"],
            "record_ids": ["fetch-1"],
            "external_writes_allowed": False,
        },
        confidence=0.94,
    )


def _patch(text: str) -> dict:
    return {
        "target_skill_id": "safe-fetch",
        "parent_skill_version": "1.0.0",
        "capability_granularity": "atomic",
        "trigger_conditions": ["one recoverable static fetch failure"],
        "applicability_preconditions": ["the URL passed public URL validation"],
        "operations": [
            {
                "kind": "add",
                "target_section": "# Safe Fetch",
                "anchor": "# Safe Fetch",
                "old_text": "",
                "new_text": text,
                "rationale": "Avoid repeating a failed route.",
            }
        ],
        "expected_improvements": ["reduce repeated route failures"],
        "risks": ["fallback may add latency"],
        "source_episode_ids": ["episode-coevo"],
        "requested_tools": [],
        "requested_permissions": [],
        "gate_bypasses": [],
    }


def _verification() -> dict:
    return {
        "decision": "accept_for_replay",
        "summary": "Locally replayable.",
        "issues": [],
        "checks": {
            "overgeneralization_safe": True,
            "preconditions_complete": True,
            "parent_compatible": True,
            "permissions_unchanged": True,
            "evidence_gates_preserved": True,
            "replayable": True,
        },
    }


def _replay(candidate_id: str, replay_id: str, *, token_cost: float) -> CandidateReplayObservation:
    return CandidateReplayObservation(
        replay_id=replay_id,
        candidate_id=candidate_id,
        source_episode_id="episode-coevo",
        stable_metrics={"fetch_route_failures": 1.0, "token_cost": 100.0},
        candidate_metrics={"fetch_route_failures": 0.0, "token_cost": token_cost},
        hard_checks={
            "schema_valid": True,
            "permissions_unchanged": True,
            "evidence_gates_preserved": True,
            "external_writes_blocked": True,
        },
    )


async def _services(tmp_path: Path):
    root = tmp_path / "skill-evolution"
    episode_store = EvolutionEpisodeStore(root)
    episode_store.write_run("run-coevo", [_episode()])
    artifact = (
        tmp_path
        / "artifacts"
        / "run-coevo"
        / "details"
        / "traces"
        / "fetch.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps([{"trace_id": "fetch-1", "status": "failed"}]),
        encoding="utf-8",
    )
    runtime = load_runtime_config(PROJECT_ROOT / "config" / "runtime.yaml")
    candidate_config = runtime.skill_evolution.candidate_generation.model_copy(
        update={"enabled": True}
    )
    coevolution_config = runtime.skill_evolution.verifier_coevolution.model_copy(
        update={"enabled": True}
    )
    candidate_generator = ScriptedLLM(
        "candidate-generator",
        [
            _patch("After one failure, use an existing safe fallback."),
            _patch(
                "After one recoverable failure, use one existing safe fallback and stop if its token cost exceeds the parent route budget."
            ),
        ],
    )
    semantic_verifier = ScriptedLLM(
        "semantic-verifier", [_verification(), _verification()]
    )
    verifier_generator = ScriptedLLM(
        "verifier-generator",
        [
            {
                "assertions": [
                    {
                        "kind": "metric_not_increase",
                        "subject": "token_cost",
                        "tolerance": 20,
                        "severity": "error",
                        "rationale": "A fallback must not add unbounded token cost.",
                    }
                ]
            }
        ],
    )
    candidate_store = CandidateStore(root)
    candidate_service = CandidateEvolutionService(
        registry=load_skill_registry(PROJECT_ROOT),
        episode_store=episode_store,
        candidate_store=candidate_store,
        artifact_root=tmp_path / "artifacts",
        generator_llm=candidate_generator,
        verifier_llm=semantic_verifier,
        config=candidate_config,
    )
    verifier_store = VerifierEvolutionStore(root)
    coevolution = VerifierCoEvolutionService(
        candidate_service=candidate_service,
        candidate_store=candidate_store,
        episode_store=episode_store,
        verifier_store=verifier_store,
        generator_llm=verifier_generator,
        config=coevolution_config,
    )
    candidate = await candidate_service.generate_for_episode("episode-coevo")
    return (
        candidate,
        candidate_service,
        coevolution,
        verifier_store,
        candidate_generator,
        semantic_verifier,
    )


@pytest.mark.asyncio
async def test_oracle_gap_evolves_verifier_and_reenters_skill_loop(tmp_path: Path):
    (
        candidate,
        _,
        coevolution,
        store,
        candidate_generator,
        semantic_verifier,
    ) = await _services(tmp_path)

    failed = coevolution.evaluate_replay(
        _replay(candidate.record.candidate_id, "replay-cost-regression", token_cost=150)
    )

    assert failed.status == "hold"
    assert failed.oracle_result.decision == "fail"
    assert failed.verifier_gap_id
    assert store.read_gap(failed.verifier_gap_id) is not None

    generated = await coevolution.generate_verifier_suite(failed.verifier_gap_id)
    assert generated.suite.version == "1.0.1"
    assert generated.suite.parent_version == "1.0.0"
    assert generated.suite.assertions[0].subject == "token_cost"
    assert generated.suite.status == "candidate"

    with pytest.raises(PermissionError, match="human approval"):
        coevolution.activate_verifier_suite(
            generated.suite.verifier_suite_id, generated.suite.version
        )
    release = coevolution.activate_verifier_suite(
        generated.suite.verifier_suite_id,
        generated.suite.version,
        human_review_approved=True,
    )
    assert release.human_review_approved is True
    assert release.stable_skill_mutation_enabled is False

    revised = await coevolution.revise_candidate_from_gap(failed.verifier_gap_id)
    assert revised.record.status == "accept_for_replay"
    assert revised.record.parent_candidate_id == candidate.record.candidate_id
    assert revised.record.source_verifier_gap_ids == [failed.verifier_gap_id]
    assert len(candidate_generator.calls) == 2
    assert len(semantic_verifier.calls) == 2

    passed = coevolution.evaluate_replay(
        _replay(revised.record.candidate_id, "replay-revised", token_cost=110)
    )
    assert passed.status == "ready_for_shadow"
    assert passed.oracle_result.decision == "pass"
    assert passed.production_loading_enabled is False


@pytest.mark.asyncio
async def test_replay_without_target_metric_is_held_without_gap(tmp_path: Path):
    candidate, _, coevolution, _, _, _ = await _services(tmp_path)
    observation = _replay(
        candidate.record.candidate_id, "replay-no-target", token_cost=100
    ).model_copy(
        update={
            "stable_metrics": {"token_cost": 100.0},
            "candidate_metrics": {"token_cost": 100.0},
        }
    )

    result = coevolution.evaluate_replay(observation)

    assert result.status == "hold"
    assert result.oracle_result.decision == "hold"
    assert result.verifier_gap_id == ""


@pytest.mark.asyncio
async def test_target_metric_at_floor_is_held_without_verifier_gap(tmp_path: Path):
    candidate, _, coevolution, _, _, _ = await _services(tmp_path)
    observation = _replay(
        candidate.record.candidate_id, "replay-no-headroom", token_cost=100
    ).model_copy(
        update={
            "stable_metrics": {"fetch_route_failures": 0.0, "token_cost": 100.0},
            "candidate_metrics": {"fetch_route_failures": 0.0, "token_cost": 100.0},
        }
    )

    result = coevolution.evaluate_replay(observation)

    assert result.status == "hold"
    assert result.oracle_result.decision == "hold"
    assert result.verifier_gap_id == ""
    target = next(
        item
        for item in result.oracle_result.checks
        if item.check_id == "target:fetch_route_failures"
    )
    assert target.status == "unknown"
    assert "metric floor" in target.summary


@pytest.mark.asyncio
async def test_verifier_generator_rejects_increased_model_calls(tmp_path: Path):
    candidate, _, coevolution, store, _, _ = await _services(tmp_path)
    failed = coevolution.evaluate_replay(
        CandidateReplayObservation(
            replay_id="replay-no-improvement",
            candidate_id=candidate.record.candidate_id,
            source_episode_id="episode-coevo",
            stable_metrics={
                "fetch_route_failures": 1.0,
                "token_cost": 100.0,
                "model_calls": 1.0,
            },
            candidate_metrics={
                "fetch_route_failures": 1.0,
                "token_cost": 100.0,
                "model_calls": 1.0,
            },
            hard_checks={
                "schema_valid": True,
                "permissions_unchanged": True,
                "evidence_gates_preserved": True,
                "external_writes_blocked": True,
            },
        )
    )
    coevolution.generator_llm.responses = [
        {
            "assertions": [
                {
                    "kind": "metric_strict_increase",
                    "subject": "model_calls",
                    "tolerance": 0,
                    "severity": "error",
                    "rationale": "More calls should improve the result.",
                }
            ]
        }
    ]

    with pytest.raises(ValueError, match="Unsafe metric direction"):
        await coevolution.generate_verifier_suite(failed.verifier_gap_id)

    assert store.find_suite_for_gap(failed.verifier_gap_id) is None
    payload = coevolution.generator_llm.calls[-1]
    assert payload["allowed_metric_assertions"]["model_calls"] == (
        "metric_not_increase"
    )
    assert payload["allowed_metric_assertions"]["fetch_route_failures"] == (
        "metric_strict_decrease"
    )


@pytest.mark.asyncio
async def test_verifier_generator_rejects_oracle_aggregate_as_hard_check(
    tmp_path: Path,
):
    candidate, _, coevolution, store, _, _ = await _services(tmp_path)
    failed = coevolution.evaluate_replay(
        _replay(candidate.record.candidate_id, "replay-cost-gap", token_cost=150)
    )
    coevolution.generator_llm.responses = [
        {
            "assertions": [
                {
                    "kind": "required_hard_check",
                    "subject": "cost_regression:token_cost",
                    "tolerance": 0,
                    "severity": "error",
                    "rationale": "Require the Oracle aggregate directly.",
                }
            ]
        }
    ]

    with pytest.raises(ValueError, match="unobserved hard check"):
        await coevolution.generate_verifier_suite(failed.verifier_gap_id)

    assert store.find_suite_for_gap(failed.verifier_gap_id) is None


def test_verifier_coevolution_is_disabled_by_default(tmp_path: Path):
    runtime = load_runtime_config(PROJECT_ROOT / "config" / "runtime.yaml")
    assert runtime.skill_evolution.verifier_coevolution.enabled is False
