from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.runtime_config import (
    VerifierCoEvolutionRuntimeConfig,
    load_runtime_config,
)
from deepresearch.service import ResearchService
from deepresearch.skill_evolution import (
    CandidateEvolutionService,
    CandidateGenerationDisabled,
    CandidateSkillMaterializer,
    CandidateStore,
    CandidateStoreConflict,
    EvolutionEpisode,
    EvolutionEpisodeStore,
    LocalMicroFixture,
    LocalReplayArmResult,
    LocalSkillReplayRunner,
    VerifierCoEvolutionService,
    VerifierEvolutionStore,
)
from deepresearch.skill_loader import load_skill_registry

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptedCandidateLLM:
    def __init__(self, model_id: str, responses: list[dict]):
        self.model_id = model_id
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def complete_json(self, system: str, user: str, *, max_tokens: int):
        payload = json.loads(user)
        self.calls.append((system, payload))
        value = self.responses.pop(0)
        return value, LLMResponse(
            json.dumps(value), input_tokens=100, output_tokens=50
        )

    async def complete_text(self, system: str, user: str, *, max_tokens: int):
        raise AssertionError("candidate workflow must use structured JSON")

    async def complete_with_tools(self, *, messages, tools, max_tokens: int):
        raise AssertionError("candidate workflow cannot call tools")


def _episode(
    *,
    episode_id: str = "episode-atomic",
    target_skill_id: str = "safe-fetch",
    parent_version: str = "1.0.0",
    granularity: str = "atomic",
    policy: str = "automatic_candidate_allowed",
) -> EvolutionEpisode:
    return EvolutionEpisode(
        episode_id=episode_id,
        run_id="run-candidate",
        task_id="task-1",
        target_skill_id=target_skill_id,
        target_skill_version=parent_version,
        runtime_layer="access" if granularity == "atomic" else "strategy",
        capability_granularity=granularity,
        candidate_policy=policy,
        trigger_signal_ids=["signal-1"],
        trigger_types=["fetch_route_failure"],
        scope={"type": "url", "value": "https://example.test/paper"},
        local_context={
            "summary": "The selected static fetch route failed.",
            "signal_context": {"error": "timeout"},
        },
        actions=[{"url": "https://example.test/paper"}],
        local_result={"severity": "warning", "observed_failure_or_low_gain": True},
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


def _patch(*, new_text: str = "After one recoverable failure, switch to an existing safe fallback.", **updates):
    value = {
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
                "new_text": new_text,
                "rationale": "Avoid repeating a failed route.",
            }
        ],
        "expected_improvements": ["reduce repeated route failures"],
        "risks": ["fallback may add latency"],
        "source_episode_ids": ["episode-atomic"],
        "requested_tools": [],
        "requested_permissions": [],
        "gate_bypasses": [],
    }
    value.update(updates)
    return value


def _verification(decision: str = "accept_for_replay", issues=None, **failed_checks):
    checks = {
        "overgeneralization_safe": True,
        "preconditions_complete": True,
        "parent_compatible": True,
        "permissions_unchanged": True,
        "evidence_gates_preserved": True,
        "replayable": True,
    }
    checks.update(failed_checks)
    return {
        "decision": decision,
        "summary": f"Verifier decision: {decision}",
        "issues": issues or [],
        "checks": checks,
    }


def _service(
    tmp_path: Path,
    generator: ScriptedCandidateLLM,
    verifier: ScriptedCandidateLLM,
    *,
    episode: EvolutionEpisode | None = None,
    config_updates: dict | None = None,
):
    registry = load_skill_registry(PROJECT_ROOT)
    root = tmp_path / "skill-evolution"
    episode_store = EvolutionEpisodeStore(root)
    selected = episode or _episode()
    episode_store.write_run(selected.run_id, [selected])
    artifact = (
        tmp_path
        / "artifacts"
        / selected.run_id
        / "details"
        / "traces"
        / "fetch.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            [
                {
                    "trace_id": "fetch-1",
                    "url": "https://example.test/paper",
                    "status": "failed",
                    "error": "timeout",
                }
            ]
        ),
        encoding="utf-8",
    )
    runtime = load_runtime_config(PROJECT_ROOT / "config" / "runtime.yaml")
    config = runtime.skill_evolution.candidate_generation.model_copy(
        update={"enabled": True, **(config_updates or {})}
    )
    candidate_store = CandidateStore(root)
    service = CandidateEvolutionService(
        registry=registry,
        episode_store=episode_store,
        candidate_store=candidate_store,
        artifact_root=tmp_path / "artifacts",
        generator_llm=generator,
        verifier_llm=verifier,
        config=config,
    )
    return service, candidate_store, registry


@pytest.mark.asyncio
async def test_candidate_is_accepted_stored_and_duplicate_episode_is_suppressed(tmp_path: Path):
    generator = ScriptedCandidateLLM("generator-model", [_patch()])
    verifier = ScriptedCandidateLLM("verifier-model", [_verification()])
    service, store, registry = _service(tmp_path, generator, verifier)

    first = await service.generate_for_episode("episode-atomic")
    second = await service.generate_for_episode("episode-atomic")

    assert first.record.status == "accept_for_replay"
    assert first.record.final_patch is not None
    assert first.record.final_patch.operations[0].kind == "add"
    assert first.record.stable_skill_mutation_enabled is False
    assert first.record.production_loading_enabled is False
    assert first.record.total_input_tokens == 200
    assert first.record.total_output_tokens == 100
    assert Path(first.store_path).is_file()
    assert store.read(first.record.candidate_id) == first.record
    assert second.reused_existing is True
    assert len(generator.calls) == 1
    assert len(verifier.calls) == 1
    assert registry.get("safe-fetch").skill_version == "1.0.0"
    assert PROJECT_ROOT / "skills" not in Path(first.store_path).parents


@pytest.mark.asyncio
async def test_scripted_generator_candidate_reaches_project_oracle(tmp_path: Path):
    added_instruction = (
        "After one recoverable failure, switch to an existing safe fallback."
    )
    generator = ScriptedCandidateLLM(
        "generator-model", [_patch(new_text=added_instruction)]
    )
    semantic_verifier = ScriptedCandidateLLM(
        "semantic-verifier-model", [_verification()]
    )
    candidate_service, candidate_store, registry = _service(
        tmp_path, generator, semantic_verifier
    )
    generated = await candidate_service.generate_for_episode("episode-atomic")
    verifier_store = VerifierEvolutionStore(tmp_path / "skill-evolution")
    replay_service = VerifierCoEvolutionService(
        candidate_service=candidate_service,
        candidate_store=candidate_store,
        episode_store=candidate_service.episode_store,
        verifier_store=verifier_store,
        generator_llm=semantic_verifier,
        config=VerifierCoEvolutionRuntimeConfig(enabled=True),
    )

    async def executor(fixture, instructions: str, arm: str):
        improved = added_instruction in instructions
        return LocalReplayArmResult(
            arm=arm,
            metrics={"fetch_route_failures": 0.0 if improved else 1.0},
            hard_checks={"ssrf_policy_preserved": True},
            output={"selected_route": "fallback" if improved else "failed-static"},
        )

    execution = await LocalSkillReplayRunner(
        materializer=CandidateSkillMaterializer(
            registry=registry,
            replay_root=tmp_path / "local-replays",
        ),
        verifier_service=replay_service,
        executor=executor,
    ).run(
        fixture=LocalMicroFixture(
            fixture_id="MICRO-E2E-fetch-route-v1",
            target_skill_id="safe-fetch",
            trigger_type="fetch_route_failure",
            input_snapshot={"url": "https://example.test/paper"},
            target_metric="fetch_route_failures",
            invariants=["ssrf_policy_preserved"],
        ),
        candidate=generated.record,
        episode=candidate_service.episode_store.get_episode("episode-atomic"),
    )

    assert len(generator.calls) == 1
    assert len(semantic_verifier.calls) == 1
    assert generated.record.status == "accept_for_replay"
    assert execution.result.status == "ready_for_shadow"
    assert execution.result.oracle_result.decision == "pass"
    assert execution.result.stable_skill_mutation_enabled is False
    assert execution.result.production_loading_enabled is False
    assert execution.result.observation.stable_metrics["fetch_route_failures"] == 1.0
    assert execution.result.observation.candidate_metrics["fetch_route_failures"] == 0.0


@pytest.mark.asyncio
async def test_deterministic_guard_rejects_permission_expansion_before_verifier(tmp_path: Path):
    generator = ScriptedCandidateLLM(
        "generator-model",
        [_patch(requested_permissions=["unrestricted_network"])],
    )
    verifier = ScriptedCandidateLLM("verifier-model", [])
    service, _, _ = _service(tmp_path, generator, verifier)

    result = await service.generate_for_episode("episode-atomic")

    assert result.record.status == "reject"
    assert not verifier.calls
    assert result.record.revisions[0].deterministic_issues[0].code == "permission_expansion"


@pytest.mark.asyncio
async def test_single_string_fields_are_not_split_into_characters(tmp_path: Path):
    payload = _patch()
    payload["trigger_conditions"] = "one recoverable failure"
    payload["applicability_preconditions"] = "public URL validation passed"
    payload["expected_improvements"] = "reduce repeated route failures"
    payload["risks"] = "fallback may add latency"
    generator = ScriptedCandidateLLM("generator-model", [payload])
    verifier = ScriptedCandidateLLM("verifier-model", [_verification()])
    service, _, _ = _service(tmp_path, generator, verifier)

    result = await service.generate_for_episode("episode-atomic")

    patch = result.record.final_patch
    assert patch is not None
    assert patch.trigger_conditions == ["one recoverable failure"]
    assert patch.applicability_preconditions == ["public URL validation passed"]
    assert patch.expected_improvements == ["reduce repeated route failures"]
    assert patch.risks == ["fallback may add latency"]


@pytest.mark.asyncio
async def test_verifier_can_request_one_bounded_revision(tmp_path: Path):
    generator = ScriptedCandidateLLM(
        "generator-model",
        [
            _patch(new_text="Always switch routes."),
            _patch(new_text="After one recoverable failure, switch to an existing safe fallback."),
        ],
    )
    verifier = ScriptedCandidateLLM(
        "verifier-model",
        [
            _verification(
                "revise",
                issues=[
                    {
                        "code": "overgeneralized",
                        "severity": "error",
                        "message": "The trigger is too broad.",
                        "suggested_fix": "Limit it to recoverable failures.",
                    }
                ],
                overgeneralization_safe=False,
            ),
            _verification(),
        ],
    )
    service, _, _ = _service(tmp_path, generator, verifier)

    result = await service.generate_for_episode("episode-atomic")

    assert result.record.status == "accept_for_replay"
    assert len(result.record.revisions) == 2
    assert result.record.revisions[1].changed_from_previous is True
    assert "verifier_issues" in generator.calls[1][1]
    assert "directly to the unchanged parent Skill" in generator.calls[1][0]
    assert len(verifier.calls) == 2


@pytest.mark.asyncio
async def test_missing_revision_identity_fields_default_to_episode(tmp_path: Path):
    revised = _patch(
        new_text=(
            "After one recoverable failure, switch to an existing safe fallback."
        )
    )
    revised["target_skill_id"] = ""
    revised["parent_skill_version"] = ""
    revised["capability_granularity"] = ""
    revised["source_episode_ids"] = []
    generator = ScriptedCandidateLLM(
        "generator-model",
        [_patch(new_text="Always switch routes."), revised],
    )
    verifier = ScriptedCandidateLLM(
        "verifier-model",
        [
            _verification(
                "revise",
                issues=[
                    {
                        "code": "overgeneralized",
                        "severity": "error",
                        "message": "The trigger is too broad.",
                        "suggested_fix": "Limit it to recoverable failures.",
                    }
                ],
                overgeneralization_safe=False,
            ),
            _verification(),
        ],
    )
    service, _, _ = _service(tmp_path, generator, verifier)

    result = await service.generate_for_episode("episode-atomic")

    patch = result.record.final_patch
    assert result.record.status == "accept_for_replay"
    assert patch is not None
    assert patch.target_skill_id == "safe-fetch"
    assert patch.parent_skill_version == "1.0.0"
    assert patch.capability_granularity == "atomic"
    assert patch.source_episode_ids == ["episode-atomic"]


@pytest.mark.asyncio
async def test_functional_episode_is_held_without_human_review(tmp_path: Path):
    functional = _episode(
        episode_id="episode-functional",
        target_skill_id="evidence-extraction",
        granularity="functional",
        policy="episode_requires_review",
    )
    generator = ScriptedCandidateLLM("generator-model", [])
    verifier = ScriptedCandidateLLM("verifier-model", [])
    service, _, _ = _service(tmp_path, generator, verifier, episode=functional)

    result = await service.generate_for_episode("episode-functional")

    assert result.record.status == "hold"
    assert "requires explicit human review" in result.record.final_reasons[0]
    assert not generator.calls
    assert not verifier.calls


@pytest.mark.asyncio
async def test_candidate_token_preflight_holds_before_exceeding_limit(tmp_path: Path):
    generator = ScriptedCandidateLLM("generator-model", [_patch()])
    verifier = ScriptedCandidateLLM("verifier-model", [_verification()])
    service, _, _ = _service(
        tmp_path,
        generator,
        verifier,
        config_updates={"max_total_tokens_per_candidate": 2000},
    )

    result = await service.generate_for_episode("episode-atomic")

    assert result.record.status == "hold"
    assert "CandidateTokenBudgetExceeded" in result.record.final_reasons[0]
    assert not generator.calls
    assert not verifier.calls


def test_candidate_store_is_immutable(tmp_path: Path):
    generator = ScriptedCandidateLLM("generator-model", [])
    verifier = ScriptedCandidateLLM("verifier-model", [])
    _, store, _ = _service(tmp_path, generator, verifier)
    record = {
        "candidate_id": "candidate-fixed",
        "source_episode_id": "episode-atomic",
        "run_id": "run-candidate",
        "target_skill_id": "safe-fetch",
        "parent_skill_version": "1.0.0",
        "capability_granularity": "atomic",
        "status": "hold",
        "final_reasons": ["fixture"],
    }
    from deepresearch.skill_evolution import CandidateRecord

    original = CandidateRecord.model_validate(record)
    store.write(original)
    store.write(original)
    with pytest.raises(CandidateStoreConflict, match="immutable"):
        store.write(original.model_copy(update={"final_reasons": ["changed"]}))


def test_candidate_generation_is_disabled_by_default(tmp_path: Path):
    generator = ScriptedCandidateLLM("generator-model", [])
    verifier = ScriptedCandidateLLM("verifier-model", [])
    service, _, _ = _service(tmp_path, generator, verifier)
    service.config = service.config.model_copy(update={"enabled": False})

    with pytest.raises(CandidateGenerationDisabled):
        __import__("asyncio").run(service.generate_for_episode("episode-atomic"))


@pytest.mark.asyncio
async def test_research_service_exposes_explicit_candidate_entrypoint(tmp_path: Path):
    root = tmp_path / "skill-evolution"
    episode = _episode()
    EvolutionEpisodeStore(root).write_run(episode.run_id, [episode])
    artifact = (
        tmp_path
        / "artifacts"
        / episode.run_id
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
    runtime = runtime.model_copy(
        update={
            "skill_evolution": runtime.skill_evolution.model_copy(
                update={
                    "store_root": str(root),
                    "candidate_generation": (
                        runtime.skill_evolution.candidate_generation.model_copy(
                            update={"enabled": True}
                        )
                    ),
                }
            )
        }
    )
    generator = ScriptedCandidateLLM("generator-model", [_patch()])
    verifier = ScriptedCandidateLLM("verifier-model", [_verification()])
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        candidate_generator_llm=generator,
        candidate_verifier_llm=verifier,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=runtime,
    )

    async with service:
        result = await service.generate_skill_candidate("episode-atomic")

    assert result.record.status == "accept_for_replay"
    assert result.record.generator_model_id == "generator-model"
    assert result.record.verifier_model_id == "verifier-model"
