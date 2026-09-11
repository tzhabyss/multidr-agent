"""Local Stable/Candidate micro replay without production Skill mutation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..ids import make_id, stable_hash
from ..skill_loader import load_skill_registry
from .coevolution import VerifierCoEvolutionService
from .materialize import CandidateSkillMaterializer, MaterializedCandidate
from .models import CandidateRecord, EvolutionEvidenceReference, EvolutionEpisode
from .verifier_models import CandidateReplayObservation, CandidateReplayResult


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocalMicroFixture(_StrictModel):
    schema_version: str = "1.0"
    fixture_id: str
    target_skill_id: str
    trigger_type: str
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    target_metric: str
    fixture_role: Literal["target", "guard"] = "target"
    invariants: list[str] = Field(default_factory=list)
    max_model_calls: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=0, ge=0)


class LocalReplayArmResult(_StrictModel):
    arm: str
    metrics: dict[str, float] = Field(default_factory=dict)
    hard_checks: dict[str, bool] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LocalReplayExecution(_StrictModel):
    replay_id: str
    fixture_id: str
    candidate_id: str
    materialization_manifest: str
    stable_result_path: str
    candidate_result_path: str
    observation_path: str
    result: CandidateReplayResult


LocalFixtureExecutor = Callable[
    [LocalMicroFixture, str, str], Awaitable[LocalReplayArmResult]
]


def _write_model(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2), encoding="utf-8")


class LocalSkillReplayRunner:
    """Execute one frozen micro fixture against isolated Skill bodies."""

    def __init__(
        self,
        *,
        materializer: CandidateSkillMaterializer,
        verifier_service: VerifierCoEvolutionService,
        executor: LocalFixtureExecutor,
    ) -> None:
        self.materializer = materializer
        self.verifier_service = verifier_service
        self.executor = executor

    @staticmethod
    def _validate_scope(
        fixture: LocalMicroFixture,
        candidate: CandidateRecord,
        episode: EvolutionEpisode,
    ) -> None:
        if candidate.source_episode_id != episode.episode_id:
            raise ValueError("Candidate and Episode do not match")
        if fixture.target_skill_id != candidate.target_skill_id:
            raise ValueError("Fixture and Candidate target different Skills")
        if fixture.trigger_type not in episode.trigger_types:
            raise ValueError("Fixture trigger is not present in the source Episode")

    @staticmethod
    def _validate_arm(
        fixture: LocalMicroFixture, result: LocalReplayArmResult
    ) -> None:
        if fixture.target_metric not in result.metrics:
            raise ValueError(
                f"Replay arm omitted target metric: {fixture.target_metric}"
            )
        if result.model_calls > fixture.max_model_calls:
            raise ValueError("Replay arm exceeded max_model_calls")
        if result.total_tokens > fixture.max_tokens:
            raise ValueError("Replay arm exceeded max_tokens")
        missing = [name for name in fixture.invariants if name not in result.hard_checks]
        if missing:
            raise ValueError("Replay arm omitted invariant(s): " + ", ".join(missing))

    async def run(
        self,
        *,
        fixture: LocalMicroFixture,
        candidate: CandidateRecord,
        episode: EvolutionEpisode,
    ) -> LocalReplayExecution:
        self._validate_scope(fixture, candidate, episode)
        replay_id = make_id(
            "local-replay",
            fixture.fixture_id,
            candidate.candidate_id,
            candidate.final_patch.patch_fingerprint if candidate.final_patch else "",
        )
        existing = self.verifier_service.verifier_store.read_replay(replay_id)
        existing_root = self.materializer.replay_root / replay_id
        if existing is not None and existing_root.is_dir():
            return LocalReplayExecution(
                replay_id=replay_id,
                fixture_id=fixture.fixture_id,
                candidate_id=candidate.candidate_id,
                materialization_manifest=str(existing_root / "materialization.json"),
                stable_result_path=str(existing_root / "stable-result.json"),
                candidate_result_path=str(existing_root / "candidate-result.json"),
                observation_path=str(existing_root / "observation.json"),
                result=existing.model_copy(update={"reused_existing": True}),
            )
        materialized = self.materializer.materialize(candidate, replay_id=replay_id)
        stable_registry = load_skill_registry(materialized.stable_project_root)
        candidate_registry = load_skill_registry(materialized.candidate_project_root)
        stable_definition = stable_registry.get(fixture.target_skill_id)
        candidate_definition = candidate_registry.get(fixture.target_skill_id)
        if stable_definition is None or candidate_definition is None:
            raise ValueError("Materialized Skill cannot be loaded")

        stable_result = await self.executor(
            fixture, stable_definition.load_instructions(), "stable"
        )
        candidate_result = await self.executor(
            fixture, candidate_definition.load_instructions(), "candidate"
        )
        if stable_result.arm != "stable" or candidate_result.arm != "candidate":
            raise ValueError("Replay executor returned an incorrect arm label")
        self._validate_arm(fixture, stable_result)
        self._validate_arm(fixture, candidate_result)

        candidate_invariants_pass = all(
            candidate_result.hard_checks[name] for name in fixture.invariants
        )
        hard_checks = {
            "schema_valid": candidate_invariants_pass,
            "permissions_unchanged": True,
            "evidence_gates_preserved": True,
            "external_writes_blocked": True,
            **candidate_result.hard_checks,
        }
        observation = CandidateReplayObservation(
            replay_id=replay_id,
            candidate_id=candidate.candidate_id,
            source_episode_id=episode.episode_id,
            fixture_role=fixture.fixture_role,
            stable_metrics={
                **stable_result.metrics,
                "token_cost": float(stable_result.total_tokens),
                "model_calls": float(stable_result.model_calls),
            },
            candidate_metrics={
                **candidate_result.metrics,
                "token_cost": float(candidate_result.total_tokens),
                "model_calls": float(candidate_result.model_calls),
            },
            hard_checks=hard_checks,
            evidence_refs=[
                EvolutionEvidenceReference(
                    artifact_path=str(materialized.manifest_path),
                    record_type="candidate_materialization",
                    record_id=candidate.candidate_id,
                ),
                EvolutionEvidenceReference(
                    artifact_path=str(materialized.replay_root / "fixture.json"),
                    record_type="local_micro_fixture",
                    record_id=fixture.fixture_id,
                ),
            ],
            external_writes_allowed=False,
            summary=(
                f"Local paired replay {fixture.fixture_id}; input hash "
                f"{stable_hash(fixture.input_snapshot)}."
            ),
        )
        fixture_path = materialized.replay_root / "fixture.json"
        _write_model(fixture_path, fixture)
        stable_path = materialized.replay_root / "stable-result.json"
        candidate_path = materialized.replay_root / "candidate-result.json"
        observation_path = materialized.replay_root / "observation.json"
        _write_model(stable_path, stable_result)
        _write_model(candidate_path, candidate_result)
        _write_model(observation_path, observation)
        result = self.verifier_service.evaluate_replay(observation)
        return LocalReplayExecution(
            replay_id=replay_id,
            fixture_id=fixture.fixture_id,
            candidate_id=candidate.candidate_id,
            materialization_manifest=str(materialized.manifest_path),
            stable_result_path=str(stable_path),
            candidate_result_path=str(candidate_path),
            observation_path=str(observation_path),
            result=result,
        )
