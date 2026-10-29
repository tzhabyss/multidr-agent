from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.artifacts import ArtifactWorkspace
from deepresearch.runtime_config import RuntimeConfig, load_runtime_config
from deepresearch.schemas import ResearchRequest
from deepresearch.service import ResearchService
from deepresearch.skill_evolution import (
    EpisodeStoreConflict,
    EvolutionEpisodeStore,
    SkillEvolutionObserver,
)
from deepresearch.skill_loader import load_skill_registry

from .fakes import FakeFetcher, FakeLLM, FakeSearchBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _registry():
    return load_skill_registry(PROJECT_ROOT)


def _state() -> dict:
    return {
        "run_id": "run-skillx-lite",
        "started_at": 1_700_000_000.0,
        "sensor_history": [
            {
                "snapshot_id": "sensor-1",
                "window_id": "r1-w1",
                "query_count": 1,
                "bound_evidence_count": 1,
                "duplicate_query_ratio": 0.0,
                "repeated_domain_ratio": 0.0,
                "repeated_source_family_failure_count": 0,
            },
            {
                "snapshot_id": "sensor-2",
                "window_id": "r1-w2",
                "query_count": 4,
                "bound_evidence_count": 1,
                "duplicate_query_ratio": 0.75,
                "repeated_domain_ratio": 0.7,
                "repeated_source_family_failure_count": 1,
            },
        ],
        "failure_memory": [],
        "fetch_traces": [
            {
                "trace_id": "fetch-failed",
                "task_id": "task-1",
                "url": "https://blocked.example/a",
                "status": "failed",
                "error": "access challenge",
                "content_chars": 0,
                "accepted_claims": 0,
            },
            {
                "trace_id": "fetch-no-binding",
                "task_id": "task-1",
                "url": "https://source.example/b",
                "status": "fetched",
                "content_chars": 5000,
                "accepted_claims": 0,
            },
        ],
        "agent_loop_traces": [
            {
                "trace_id": "loop-1",
                "task_id": "task-1",
                "iteration": 2,
                "observations": [
                    {
                        "name": "web_search",
                        "success": False,
                        "error": "provider unavailable",
                    }
                ],
            }
        ],
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "Needs another source.",
                "status": "needs_verification",
            }
        ],
        "links": [
            {
                "link_id": "link-1",
                "claim_id": "claim-2",
                "evidence_id": "evidence-2",
                "entailment_status": "rejected",
            }
        ],
        "evidence_bindings": [],
    }


def test_skill_metadata_keeps_runtime_layer_and_capability_granularity_separate():
    registry = _registry()
    web = registry.get("web-search")
    extraction = registry.get("evidence-extraction")
    planning = registry.get("research-planning")

    assert web is not None
    assert (web.skill_layer, web.capability_granularity) == ("access", "atomic")
    assert web.evolution_candidate_policy == "automatic_candidate_allowed"
    assert extraction is not None
    assert (extraction.skill_layer, extraction.capability_granularity) == (
        "strategy",
        "functional",
    )
    assert extraction.evolution_candidate_policy == "episode_requires_review"
    assert planning is not None
    assert planning.evolution_candidate_policy == "observe_only"
    snapshot = {item["skill_id"]: item for item in registry.loading_snapshot()}
    assert snapshot["web-search"]["skill_version"] == "1.0.0"
    assert snapshot["web-search"]["capability_granularity"] == "atomic"


def test_observer_normalizes_existing_runtime_signals_and_builds_replayable_episodes():
    observer = SkillEvolutionObserver(_registry())
    signals, episodes = observer.observe(_state())
    types = {item.signal_type for item in signals}

    assert {
        "consecutive_no_evidence_gain",
        "duplicate_query",
        "repeated_source_domain",
        "source_family_repeated_failure",
        "fetch_route_failure",
        "extraction_without_binding",
        "skill_execution_failure",
        "weak_or_rejected_claim",
    }.issubset(types)
    assert len({item.signal_id for item in signals}) == len(signals)
    assert len(episodes) == len(signals)
    assert all(item.evidence_refs for item in episodes)
    assert all(item.replay_spec["external_writes_allowed"] is False for item in episodes)

    web_episode = next(
        item
        for item in episodes
        if item.target_skill_id == "web-search"
        and item.trigger_types == ["duplicate_query"]
    )
    assert web_episode.capability_granularity == "atomic"
    assert web_episode.candidate_policy == "automatic_candidate_allowed"

    extraction_episode = next(
        item
        for item in episodes
        if item.target_skill_id == "evidence-extraction"
    )
    assert extraction_episode.capability_granularity == "functional"
    assert extraction_episode.candidate_policy == "episode_requires_review"


def test_episode_store_is_isolated_idempotent_and_immutable(tmp_path: Path):
    observer = SkillEvolutionObserver(_registry())
    _, episodes = observer.observe(_state())
    store = EvolutionEpisodeStore(tmp_path / ".runtime" / "skill-evolution")

    first = store.write_run("run-skillx-lite", episodes)
    second = store.write_run("run-skillx-lite", episodes)

    assert first == second
    assert store.read_run("run-skillx-lite") == episodes
    assert store.query(target_skill_id="web-search")
    assert all(
        item.target_skill_id == "web-search"
        for item in store.query(target_skill_id="web-search")
    )
    assert store.query(trigger_type="skill_execution_failure", limit=1)[0].trigger_types == [
        "skill_execution_failure"
    ]
    assert store.episodes_root != store.candidates_root
    assert PROJECT_ROOT / "skills" not in first.parents

    changed = episodes[0].model_copy(update={"confidence": 0.01})
    with pytest.raises(EpisodeStoreConflict, match="immutable"):
        store.write_run("run-skillx-lite", [changed, *episodes[1:]])


def test_runtime_default_is_off_and_project_config_observes_general_research():
    default = RuntimeConfig.model_validate(
        {
            "llm_output_tokens": {
                "planner": 1024,
                "worker_strategy": 512,
                "worker_reflection": 512,
                "worker_agent_loop": 512,
                "evidence_extraction_min": 256,
                "evidence_extraction_max": 512,
                "synthesizer_assessment": 512,
                "report_writer": 1024,
                "reviewer": 512,
                "extraction_context": 2048,
            }
        }
    )
    configured = load_runtime_config(PROJECT_ROOT / "config" / "runtime.yaml")

    assert default.skill_evolution.mode == "off"
    assert configured.skill_evolution.mode == "observe"
    assert configured.effective_skill_evolution_mode("general_research") == "observe"
    assert configured.effective_skill_evolution_mode("drb_evaluation") == "off"


def test_skill_evolution_is_only_added_to_audit_when_observation_ran():
    base = {
        "run_id": "run-audit",
        "request": {},
        "contributions": [],
    }
    disabled = ArtifactWorkspace._audit_log(base)
    enabled = ArtifactWorkspace._audit_log(
        {
            **base,
            "skill_evolution_observation": {"mode": "observe", "status": "completed"},
            "evolution_signals": [{"signal_id": "signal-1"}],
            "evolution_episodes": [{"episode_id": "episode-1"}],
        }
    )

    assert "skill_evolution" not in disabled
    assert enabled["skill_evolution"]["observation"]["mode"] == "observe"
    assert enabled["skill_evolution"]["signals"][0]["signal_id"] == "signal-1"


@pytest.mark.asyncio
async def test_observe_mode_is_an_isolated_end_to_end_side_path(tmp_path: Path):
    config = load_runtime_config(PROJECT_ROOT / "config" / "runtime.yaml")
    # This observes FakeFetcher/FakeLLM artifacts; it must not initialize the
    # deployment's optional external knowledge database.
    config.knowledge.enabled = False
    config = config.model_copy(
        update={
            "skill_evolution": config.skill_evolution.model_copy(
                update={
                    "mode": "observe",
                    "store_root": str(tmp_path / "skill-evolution"),
                }
            )
        }
    )
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    )

    async with service:
        result = await service.run(
            ResearchRequest(
                question="Verify a public fact",
                run_id="run-skillx-observe",
                options={"max_rounds": 1, "max_worker_iterations": 1},
            )
        )

    artifact_dir = Path(result.artifact_dir)
    observation = artifact_dir / "details" / "skill-evolution" / "observation.json"
    episodes = artifact_dir / "details" / "skill-evolution" / "episodes.json"
    central = tmp_path / "skill-evolution" / "episodes" / "run-skillx-observe.json"

    assert observation.exists()
    assert episodes.exists()
    assert central.exists()
    assert not (tmp_path / "skill-evolution" / "candidates").exists()


@pytest.mark.asyncio
async def test_evaluation_profile_disables_skill_evolution_observation(
    tmp_path: Path,
):
    config = load_runtime_config(PROJECT_ROOT / "config" / "runtime.yaml")
    config = config.model_copy(
        update={
            "skill_evolution": config.skill_evolution.model_copy(
                update={"store_root": str(tmp_path / "skill-evolution")}
            )
        }
    )
    service = ResearchService.in_memory(
        llm=FakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    )

    state = {
        "request": ResearchRequest(
            question="Evaluate a public fact",
            profile_id="drb_evaluation",
        ).model_dump(mode="json")
    }
    async with service:
        assert service._graph is not None
        assert await service._graph._collect_skill_evolution(state) == {}
    assert not (tmp_path / "skill-evolution").exists()
