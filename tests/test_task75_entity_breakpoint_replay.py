from __future__ import annotations

import json
from pathlib import Path

from tools.replay_task75_entity_breakpoint import ROOT, replay_breakpoint


def test_task75_checkpoint_replay_releases_gate_and_commits_entity():
    fixture_path = (
        ROOT
        / "evaluation/fixtures/skill_evolution/"
        "task75-entity-breakpoint-replay-20260905-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    result = replay_breakpoint(fixture)

    assert result["status"] == "pass"
    assert result["scope"] == "checkpoint_only_no_search_fetch_or_llm"
    assert result["historical_breakpoint"] == {
        "task_id": "task_c340a7baaedd01b793b2e55c",
        "status": "blocked",
        "stop_reason": "missing_required_artifact_roles",
        "missing_artifact_roles": ["navigation_only"],
    }
    assert result["observed"]["target_entity_lead_count"] == 3
    assert result["observed"]["matched_downstream_target_count"] == 4
    assert result["observed"]["pre_verification_entity_count"] == 0
    assert result["observed"]["post_verification_entity_count"] == 1
    assert result["observed"]["post_verification_scope_status"] == "open"
    assert result["failed_checks"] == []
    assert all(result["checks"].values())
    assert result["token_usage"] == 0
