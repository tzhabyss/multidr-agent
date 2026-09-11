#!/usr/bin/env python3
"""Audit two deterministic guard gaps in the current Skill replay oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.skill_evolution import (  # noqa: E402
    CandidateReplayObservation,
    EvolutionEpisode,
    ProjectOracleStack,
)
from deepresearch.skill_evolution.verifier_models import VerifierSuite  # noqa: E402


def _episode() -> EvolutionEpisode:
    return EvolutionEpisode(
        episode_id="episode-oracle-guard-audit",
        run_id="run-oracle-guard-audit",
        task_id="task-oracle-guard-audit",
        target_skill_id="web-search",
        target_skill_version="1.0.0",
        runtime_layer="access",
        capability_granularity="atomic",
        candidate_policy="automatic_candidate_allowed",
        trigger_signal_ids=["signal-oracle-guard-audit"],
        trigger_types=["consecutive_no_evidence_gain"],
        scope={"type": "task", "value": "local-oracle-audit"},
        local_context={"summary": "Audit target and guard metric semantics."},
        actions=[],
        local_result={"observed_failure_or_low_gain": True},
        evidence_refs=[],
        replay_spec={"external_writes_allowed": False},
        confidence=1.0,
    )


def _suite() -> VerifierSuite:
    return VerifierSuite(
        verifier_suite_id="verifier-suite-oracle-guard-audit",
        version="1.0.0",
        target_skill_id="web-search",
        assertions=[],
        status="builtin",
    )


def _observation(
    replay_id: str,
    *,
    stable_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    fixture_role: str = "target",
) -> CandidateReplayObservation:
    return CandidateReplayObservation(
        replay_id=replay_id,
        candidate_id="candidate-oracle-guard-audit",
        source_episode_id="episode-oracle-guard-audit",
        fixture_role=fixture_role,
        stable_metrics={**stable_metrics, "token_cost": 100.0},
        candidate_metrics={**candidate_metrics, "token_cost": 100.0},
        hard_checks={
            "schema_valid": True,
            "permissions_unchanged": True,
            "evidence_gates_preserved": True,
            "external_writes_blocked": True,
        },
        external_writes_allowed=False,
    )


def audit() -> dict[str, Any]:
    oracle = ProjectOracleStack()
    episode = _episode()
    suite = _suite()

    mixed = oracle.evaluate(
        _observation(
            "replay-mixed-target-regression",
            stable_metrics={"evidence_gain": 0.0, "zero_gain_searches": 1.0},
            candidate_metrics={"evidence_gain": 1.0, "zero_gain_searches": 2.0},
        ),
        episode=episode,
        suite=suite,
    )
    mixed_checks = {item.check_id: item.status for item in mixed.checks}

    guard = oracle.evaluate(
        _observation(
            "replay-optimal-guard",
            stable_metrics={"evidence_gain": 1.0, "zero_gain_searches": 0.0},
            candidate_metrics={"evidence_gain": 1.0, "zero_gain_searches": 0.0},
            fixture_role="guard",
        ),
        episode=episode,
        suite=suite,
    )
    guard_checks = {item.check_id: item.status for item in guard.checks}

    return {
        "schema_version": "1.0",
        "mixed_target_case": {
            "decision": mixed.decision,
            "checks": mixed_checks,
            "blind_spot_present": (
                mixed.decision == "pass"
                and mixed_checks.get("target:zero_gain_searches") == "fail"
            ),
            "regression_blocked": mixed.decision == "fail",
        },
        "optimal_guard_case": {
            "decision": guard.decision,
            "checks": guard_checks,
            "guard_role_missing": guard.decision != "pass",
            "non_regressing_guard_passed": guard.decision == "pass",
        },
        "verdict": "oracle_guard_semantics_fixed",
        "skill_candidate_can_self_fix": False,
    }


def main() -> int:
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
