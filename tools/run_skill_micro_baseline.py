#!/usr/bin/env python3
"""Run five deterministic local pain-point checks against the current Stable code."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.agents import _expand_queries
from deepresearch.failure_memory import active_failures, failures_from_fetch_traces
from deepresearch.graph import ResearchGraph
from deepresearch.observation_middleware import observations_from_extraction
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import (
    EvidenceExtractionResult,
    EvidenceRecord,
    ResearchTask,
    SearchSchemaDefinition,
    SourceRecord,
    TaskBudget,
)
from deepresearch.skill_loader import load_skill_registry
from deepresearch.skill_evolution import (
    CandidateRecord,
    CandidateSkillMaterializer,
    EvolutionEpisode,
    LocalMicroFixture,
    LocalSkillReplayRunner,
)
from deepresearch.time_utils import now


def _result(
    fixture_id: str,
    status: str,
    *,
    target: str,
    observed: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "status": status,
        "target": target,
        "observed": observed,
        "reason": reason,
    }


def micro_01_duplicate_query() -> dict[str, Any]:
    queries = [
        "SkillX benchmark",
        "SkillX benchmark",
        "SkillX GitHub repository",
    ]
    expanded = _expand_queries(
        "Research SkillX",
        "paper evidence",
        queries,
        minimum=1,
    )
    duplicates = len(expanded) - len({item.casefold() for item in expanded})
    passed = len(expanded) == 2 and duplicates == 0
    return _result(
        "MICRO-01-exact-duplicate-v1",
        "pass" if passed else "fail",
        target="Remove an exact duplicate without dropping the distinct query.",
        observed={
            "input_query_count": len(queries),
            "output_query_count": len(expanded),
            "duplicate_queries": duplicates,
            "output_queries": expanded,
        },
        reason=(
            "Current Stable removes exact case-insensitive duplicates."
            if passed
            else "Current Stable did not preserve the expected unique query set."
        ),
    )


def micro_03_failure_memory() -> dict[str, Any]:
    url = "https://fixture.invalid/report"
    records = failures_from_fetch_traces(
        run_id="micro-run",
        task_id="task-a",
        traces=[
            {
                "url": url,
                "provider": "fixture",
                "status": "failed",
                "error": "HTTP 429",
            }
        ],
        enhanced=True,
    )
    active_same_task = bool(active_failures(records, task_id="task-a"))
    active_other_task = bool(active_failures(records, task_id="task-b"))
    expired = records[0].model_copy(
        update={"expires_at": (now() - timedelta(seconds=1)).isoformat()}
    )
    inactive_after_ttl = not active_failures([expired], task_id="task-a")
    resolved = failures_from_fetch_traces(
        run_id="micro-run",
        task_id="task-a",
        traces=[
            {
                "url": url,
                "provider": "fixture",
                "status": "fetched",
                "content_chars": 1000,
                "accepted_claims": 1,
            }
        ],
        existing=records,
        enhanced=True,
    )
    inactive_after_success = bool(resolved) and not active_failures(resolved)
    passed = (
        active_same_task
        and not active_other_task
        and inactive_after_ttl
        and inactive_after_success
    )
    return _result(
        "MICRO-03-429-ttl-v1",
        "pass" if passed else "fail",
        target="Keep a 429 scoped and expiring, then retire it after success.",
        observed={
            "active_same_task": active_same_task,
            "active_other_task": active_other_task,
            "inactive_after_ttl": inactive_after_ttl,
            "inactive_after_success": inactive_after_success,
            "failure_type": records[0].failure_type,
            "expires_at_recorded": bool(records[0].expires_at),
        },
        reason=(
            "Current Stable scopes the failure, records TTL, and retires it after success."
            if passed
            else "Failure Memory did not satisfy scope, TTL, and recovery invariants."
        ),
    )


def micro_06_binding_recovery() -> dict[str, Any]:
    task = ResearchTask(
        task_id="micro-binding",
        objective="Fill Entity-1 sample size",
        research_dimension="sample size",
        research_object="Entity-1",
        target_entity_ids=["entity-1"],
        target_field_keys=["sample_size"],
        requirement_ids=["req-sample-size"],
        produces_requirement_ids=["req-sample-size"],
        queries=["Entity-1 sample size"],
        budget=TaskBudget(max_searches=0, max_fetches=0, max_wall_time_seconds=30),
    )
    content = "Entity-1 enrolled 42 participants."
    source = SourceRecord(
        source_id="src-binding",
        source_kind="web_page",
        title="Fixture source",
        canonical_uri="https://fixture.invalid/binding",
        provider="fixture",
        fetched_at="2026-09-04T00:00:00+08:00",
        fetch_status="full",
        http_status=200,
        mime_type="text/plain",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        quality_tier=1,
        quality_reason="frozen fixture",
    )
    evidence = EvidenceRecord(
        evidence_id="ev-binding",
        source_id=source.source_id,
        quote=content,
        quote_hash=hashlib.sha256(content.encode()).hexdigest(),
        location={"kind": "text", "value": "0"},
        requirement_ids=["req-sample-size"],
    )
    parsed = EvidenceExtractionResult.model_validate(
        {
            "artifact_role": "field_evidence",
            "target_entity_type": "study",
            "claims": [
                {
                    "text": "Entity-1 enrolled 42 participants.",
                    "quote": content,
                    "requirement_ids": ["req-sample-size"],
                    "field_value": 42,
                }
            ],
        }
    )
    schema = SearchSchemaDefinition.model_validate(
        {
            "schema_id": "micro-binding-schema",
            "tables": [
                {
                    "table_id": "studies",
                    "row_entity_type": "study",
                    "fields": [
                        {
                            "field_key": "sample_size",
                            "label": "Sample size",
                            "required": True,
                        }
                    ],
                }
            ],
            "scope_contract": {"target_description": "the single assigned study"},
        }
    )
    output = observations_from_extraction(
        run_id="micro-run",
        task=task,
        source=source,
        parsed=parsed,
        accepted_candidates=[parsed.claims[0].model_dump()],
        evidence=[evidence],
        schema=schema,
    )
    binding_count = len(output["evidence_bindings"])
    passed = binding_count == 1
    return _result(
        "MICRO-06-single-target-binding-v1",
        "pass" if passed else "fail",
        target=(
            "Recover a binding when the task has one entity and one field but the "
            "extractor omitted entity_id/field_key metadata."
        ),
        observed={
            "task_target_entity_ids": task.target_entity_ids,
            "task_target_field_keys": task.target_field_keys,
            "binding_count": binding_count,
            "bindings": output["evidence_bindings"],
        },
        reason=(
            "Current Stable recovered the unambiguous task-level binding."
            if passed
            else "Current Stable only binds extractor-supplied entity_id/field_key and does not recover this unambiguous task-level binding."
        ),
    )


async def micro_07_required_elements() -> dict[str, Any]:
    state = {
        "run_id": "micro-elements",
        "round_no": 1,
        "wave_no": 1,
        "wave_history": [{"wave_id": "r1-w1"}],
        "current_plan": {
            "requirements": [
                {
                    "requirement_id": "req-legal",
                    "text": "Explain the legal regime.",
                    "required_elements": [
                        "legal basis",
                        "liability type",
                        "burden of proof",
                    ],
                }
            ]
        },
        "current_tasks": [
            {
                "task_id": "task-legal",
                "requirement_ids": ["req-legal"],
                "produces_requirement_ids": ["req-legal"],
            }
        ],
        "task_runtime": [{"task_id": "task-legal", "status": "completed"}],
        "contributions": [
            {
                "task_id": "task-legal",
                "status": "completed",
                "usable": True,
                "sufficiency_passed": True,
            }
        ],
        "sources": [
            {
                "source_id": "src-legal",
                "canonical_uri": "https://fixture.invalid/legal",
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev-legal",
                "source_id": "src-legal",
                "requirement_ids": ["req-legal"],
            }
        ],
        "claims": [
            {
                "claim_id": "claim-legal-basis",
                "text": "The legal basis is statute X.",
                "requirement_ids": ["req-legal"],
            }
        ],
        "links": [
            {
                "link_id": "link-legal",
                "claim_id": "claim-legal-basis",
                "evidence_id": "ev-legal",
                "relation": "supports",
                "entailment_status": "verified",
            }
        ],
    }
    output = await ResearchGraph._deterministic_checks(None, state)
    status = output["requirement_status"][0]
    element_statuses = [
        item["status"] for item in status["required_element_statuses"]
    ]
    passed = (
        status["status"] == "partial"
        and status.get("operational_status") == "sufficient"
        and status.get("semantic_status") == "unassessed"
        and element_statuses == ["unassessed", "unassessed", "unassessed"]
    )
    return _result(
        "MICRO-07-three-required-elements-v1",
        "pass" if passed else "fail",
        target=(
            "Do not report semantic sufficiency before all required elements "
            "have been assessed."
        ),
        observed={
            "requirement_status": status["status"],
            "semantic_completeness_assessed": status[
                "semantic_completeness_assessed"
            ],
            "required_element_statuses": status["required_element_statuses"],
        },
        reason=(
            "Current Stable separates producer completion from unassessed semantic coverage."
            if passed
            else "Current Stable still collapses producer completion into semantic sufficiency."
        ),
    )


def micro_12_skill_routing() -> dict[str, Any]:
    runtime = load_runtime_config(ROOT / "config" / "runtime.yaml")
    registry = load_skill_registry(
        ROOT,
        skill_mode=runtime.graph_engineering.skill_mode,
        layer_settings=runtime.skill_layers.model_dump(mode="json"),
    )
    catalog = registry.prompt_instructions(
        role="worker",
        profile_id="general_research",
        phase="worker_strategy",
    )
    selected = registry.prompt_instructions(
        role="worker",
        profile_id="general_research",
        phase="worker_loop",
        selected_skills=("research-worker",),
    )
    plumbing_ok = "research-worker" in catalog and "# Research Worker" in selected
    episode = EvolutionEpisode(
        episode_id="micro-routing-episode",
        run_id="micro-run",
        target_skill_id="web-search",
        target_skill_version="1.0.0",
        runtime_layer="access",
        capability_granularity="atomic",
        candidate_policy="automatic_candidate_allowed",
        trigger_signal_ids=["micro-routing-signal"],
        trigger_types=["duplicate_query"],
        confidence=1.0,
    )
    candidate = CandidateRecord.model_validate(
        {
            "candidate_id": "micro-routing-candidate",
            "source_episode_id": episode.episode_id,
            "run_id": episode.run_id,
            "target_skill_id": "web-search",
            "parent_skill_version": "1.0.0",
            "capability_granularity": "atomic",
            "status": "accept_for_replay",
            "final_patch": {
                "candidate_id": "micro-routing-candidate",
                "target_skill_id": "web-search",
                "parent_skill_version": "1.0.0",
                "capability_granularity": "atomic",
                "trigger_conditions": ["duplicate_query"],
                "applicability_preconditions": ["same query constraints"],
                "operations": [
                    {
                        "operation_id": "micro-routing-op",
                        "kind": "add",
                        "target_section": "Web Search",
                        "anchor": "# Web Search",
                        "new_text": "MICRO_CANDIDATE_ROUTING_MARKER",
                        "rationale": "verify isolated loading",
                    }
                ],
                "expected_improvements": ["route matching Candidate"],
                "risks": ["must remain isolated"],
                "source_episode_ids": [episode.episode_id],
                "patch_fingerprint": "micro-routing-v1",
            },
        }
    )
    fixture = LocalMicroFixture(
        fixture_id="MICRO-12-candidate-routing-v1",
        target_skill_id="web-search",
        trigger_type="duplicate_query",
        target_metric="duplicate_queries",
    )
    LocalSkillReplayRunner._validate_scope(fixture, candidate, episode)
    source_path = ROOT / "skills" / "web-search" / "SKILL.md"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="dr-candidate-routing-") as temp_dir:
        materialized = CandidateSkillMaterializer(
            registry=load_skill_registry(ROOT),
            replay_root=Path(temp_dir),
        ).materialize(candidate, replay_id="micro-routing-replay")
        candidate_registry = load_skill_registry(materialized.candidate_project_root)
        candidate_definition = candidate_registry.get("web-search")
        candidate_loading_available = bool(
            candidate_definition
            and "MICRO_CANDIDATE_ROUTING_MARKER"
            in candidate_definition.load_instructions()
        )
        stable_is_unchanged = (
            hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
            and "MICRO_CANDIDATE_ROUTING_MARKER"
            not in materialized.stable_skill_path.read_text(encoding="utf-8")
        )
    trigger_matched = fixture.trigger_type in episode.trigger_types
    status = (
        "pass"
        if plumbing_ok
        and candidate_loading_available
        and stable_is_unchanged
        and trigger_matched
        else "fail"
    )
    return _result(
        "MICRO-12-candidate-routing-v1",
        status,
        target="Select and inject an isolated Candidate only for a matching trigger.",
        observed={
            "layered_catalog_available": "research-worker" in catalog,
            "selected_stable_body_loaded": "# Research Worker" in selected,
            "candidate_loading_available": candidate_loading_available,
            "candidate_isolated_from_stable": stable_is_unchanged,
            "trigger_matched": trigger_matched,
            "candidate_generation_enabled": runtime.skill_evolution.candidate_generation.enabled,
        },
        reason=(
            "A trigger-matched Candidate loads from an isolated replay copy while Stable remains unchanged."
            if status == "pass"
            else "Candidate replay routing or isolation prerequisites are incomplete."
        ),
    )


async def run() -> dict[str, Any]:
    results = [
        micro_01_duplicate_query(),
        micro_03_failure_memory(),
        micro_06_binding_recovery(),
        await micro_07_required_elements(),
        micro_12_skill_routing(),
    ]
    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("pass", "fail", "partial")
    }
    return {
        "schema_version": "1.0",
        "evaluation_type": "skill_evolution_local_preflight",
        "full_task_runs": 0,
        "network_calls": 0,
        "model_calls": 0,
        "token_usage": 0,
        "summary": counts,
        "results": results,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Skill 自进化五项局部基线测试",
        "",
        (
            "本轮只运行局部能力检查；没有完整研究任务、网络或模型调用。"
            "仅 MICRO-12 在临时目录加载隔离 Candidate，生产加载保持关闭。"
        ),
        "",
        f"- 完整任务：{payload['full_task_runs']}",
        f"- 网络调用：{payload['network_calls']}",
        f"- 模型调用：{payload['model_calls']}",
        f"- Token：{payload['token_usage']}",
        f"- 结果：{payload['summary']['pass']} pass / {payload['summary']['fail']} fail / {payload['summary']['partial']} partial",
        "",
        "| Fixture | 结果 | 目标 | 观察摘要 |",
        "|---|---|---|---|",
    ]
    for item in payload["results"]:
        observed = json.dumps(item["observed"], ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| {item['fixture_id']} | {item['status']} | {item['target']} | {observed} |"
        )
    lines.extend(["", "## 解释", ""])
    for item in payload["results"]:
        lines.extend(
            [
                f"### {item['fixture_id']} — {item['status']}",
                "",
                item["reason"],
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "skill-micro-eval" / "baseline-20260904-v1",
    )
    args = parser.parse_args()
    payload = asyncio.run(run())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    # Fixture failures are evaluation outcomes, not runner failures. A nonzero
    # exit is reserved for malformed fixtures or execution errors.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
