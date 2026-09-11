#!/usr/bin/env python3
"""Run one real-model, offline Stable/Candidate Skill micro campaign."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.llm import LLMResponse
from deepresearch.model_routing import ModelRouter
from deepresearch.runtime_config import (
    VerifierCoEvolutionRuntimeConfig,
    load_runtime_config,
)
from deepresearch.skill_evolution import (
    CandidateEvolutionService,
    CandidateGenerationResult,
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
from deepresearch.skill_evolution.models import EvolutionEvidenceReference


_QUERY_SELECTOR_SYSTEM = """You are executing a bounded Web Search Skill replay.
Choose which supplied query IDs should be executed. Follow the supplied Skill
instructions. Remove queries that would retrieve the same intent, but preserve
different entities, years, locales, fields, and source constraints. Do not add
queries and do not browse. Return one JSON object with selected_query_ids and a
short rationale."""

_ROUTE_SELECTOR_SYSTEM = """You are executing a bounded Safe Fetch Skill replay.
The input contains prior failed source families and an allowlisted set of
available route IDs with simulated outcomes. Choose an ordered list of next
route IDs. Follow the supplied Skill instructions, do not invent routes, do not
browse, and never choose a blocked route. Stop after the first successful
route. Return one JSON object with selected_route_ids and a short rationale."""

_ROUTE_HISTORY_SELECTOR_SYSTEM = """You are executing a bounded Safe Fetch
Skill replay. Review the supplied failed access attempts and choose an ordered
list of next route IDs from the allowlisted inventory. Follow the supplied Skill
instructions, do not invent routes, do not browse, and never choose a blocked
route. Stop after the first successful route. Return one JSON object with
selected_route_ids and a short rationale."""

_SOURCE_SELECTOR_SYSTEM = """You are executing a bounded Web Search Skill replay.
The input contains domains already represented in the current task and an
allowlisted candidate-source inventory. Choose exactly selection_count source
IDs to fetch next. Follow the supplied Skill instructions, preserve at least
one authoritative source, do not invent sources, and do not browse. Return one
JSON object with selected_source_ids and a short rationale."""

_REFORMULATION_SYSTEM = """You are executing a bounded Web Search Skill replay.
The previous queries produced no usable evidence. Choose exactly one supplied
query ID for the next search. Follow the supplied Skill instructions, preserve
the research scope, do not invent queries, and do not browse. Return one JSON
object with selected_query_ids and a short rationale."""

_ENTITY_MATERIALIZATION_SYSTEM = """You are executing a bounded Research Planning
Skill replay. Identify which supplied candidate entity IDs are rows in the
user's requested result set. A source document can be only a container that
lists targets, or it can itself be a target when the user asks to compare
documents. Follow the supplied Skill instructions and literal research scope.
Do not invent entities and do not browse. Return one JSON object with
selected_entity_ids and a short rationale."""

_ENTITY_DISCOVERY_SYSTEM = """You are executing a bounded Research Planning Skill
replay. Starting only from the supplied raw source excerpts, generate the
complete set of entities that belong in the user's requested result rows.
Resolve aliases and repeated mentions into one entity. Exclude source
containers, excluded/non-qualifying items, substudies, and contextual entities
unless the literal scope makes them targets. Separately classify whether the
target universe is open, stabilizing, provisionally_complete, audited_complete,
or unreachable. Do not browse. Return one JSON object with an entities array and
scope_completeness_status; every entity must contain canonical_name and may
contain aliases and source_ids. source_ids should identify supplied documents
that directly support including that entity in the requested result set.
Include a short rationale."""

_ENTITY_DISCOVERY_NEUTRAL_SYSTEM = """You are executing a bounded Research
Planning Skill replay. From the supplied research scope and source excerpts,
return one JSON object with an entities array, scope_completeness_status, and a
short rationale. Every entity must contain canonical_name and may contain
aliases and source_ids. Use only the supplied text, do not invent entities, and
do not browse. scope_completeness_status must be exactly one of open,
stabilizing, provisionally_complete, audited_complete, or unreachable."""

_FIELD_MATRIX_SYSTEM = """You are executing a bounded Evidence Extraction Skill
replay. Fill every requested entity_id x field_key cell from the supplied raw
source excerpts. Return one JSON object with a cells array and a short
rationale. Every cell must contain entity_id, field_key, status, and values.
status is supported, conflict, or nr. values is empty for nr; otherwise every
value item must contain value, source_id, and an exact contiguous quote copied
from that source. Use conflict only when equally applicable authoritative
sources disagree without a stated correction. Prefer final observed results
over planned or analytic-population values when the field asks for randomized
enrollment. Never omit a row because one field is unavailable, never invent a
value, and do not browse."""

_FIELD_MATRIX_NEUTRAL_SYSTEM = """You are executing a bounded Evidence
Extraction Skill replay. Fill every requested entity_id x field_key cell from
the supplied raw source excerpts. Return one JSON object with a cells array and
a short rationale. Every cell must contain entity_id, field_key, status, and
values. status must be exactly supported, conflict, or nr. values must be empty
for nr; otherwise every value item must contain value, source_id, and an exact
contiguous quote copied from that source. Do not invent values and do not
browse."""

_TASK_CONTRACT_SYSTEM = """You are executing a bounded Research Planning Skill
replay. Given an overall research scope, current coverage, and one already-fetched
source excerpt, produce exactly one next Worker task contract. Follow the supplied
Skill instructions. Do not extract the answer and do not browse. The contract must
state its evidence boundary instead of claiming work that this source cannot do.
Return one JSON object with task_contract and a short rationale. task_contract must
contain objective, completion_contract, source_ids, target_entity_ids,
target_field_keys, allow_partial_results, accept_unresolved_fields, and queries.
completion_contract must be exactly extract_entities or fill_fields. queries means
external search-engine query strings, not questions for the extractor; because the
source is already fetched, return an empty queries array."""

_COVERAGE_GAP_PLANNING_SYSTEM = """You are executing a bounded Research
Planning Skill replay. Compare the literal research scope, required coverage
cells, and current evidence coverage. Select every required cell that still
needs a next Worker task, while avoiding cells already supported by evidence.
Do not research, answer the underlying question, or invent cell IDs. Return one
JSON object with selected_cell_keys, scope_completeness_status, and a short
rationale. scope_completeness_status must be exactly open, stabilizing,
provisionally_complete, audited_complete, or unreachable."""

_ANALYSIS_CHAIN_SYSTEM = """You are executing a bounded Research Synthesis
Skill replay. Answer each supplied analysis request as a separate reasoning
unit grounded only in the supplied evidence items. Preserve legal terms and
explicitly connect premises to the conclusion; do not merely summarize the
evidence. Do not invent evidence IDs or browse. Return one JSON object with an
analysis_units array and a short rationale. Every unit must contain request_id,
premise_evidence_ids, reasoning, and conclusion."""


@dataclass
class CampaignBudget:
    cap: int
    spent: int = 0

    def charge_existing(self, tokens: int) -> None:
        if self.spent + tokens > self.cap:
            raise RuntimeError("Campaign token cap exceeded by Candidate generation")
        self.spent += tokens

    def preflight(self, *, system: str, user: str, max_tokens: int) -> None:
        reservation = (
            len(system.encode("utf-8"))
            + len(user.encode("utf-8"))
            + max_tokens
            + 4096
        )
        if self.spent + reservation > self.cap:
            raise RuntimeError(
                f"Campaign token preflight failed: spent={self.spent}, "
                f"next_reservation={reservation}, cap={self.cap}"
            )

    def settle(self, response: LLMResponse) -> None:
        self.charge_existing(response.input_tokens + response.output_tokens)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists() and path.read_text(encoding="utf-8") != data:
        raise RuntimeError(f"Frozen campaign artifact conflict: {path}")
    path.write_text(data, encoding="utf-8")


def _markdown(payload: dict[str, Any]) -> str:
    candidate = payload["candidate"]
    candidate_label = candidate["candidate_id"] or "未生成"
    lines = [
        "# 真实模型 Skill 局部自进化测试",
        "",
        f"- Campaign：`{payload['campaign_id']}`",
        f"- Candidate：`{candidate_label}`",
        f"- Candidate 状态：`{candidate['status']}`",
        f"- 实际 Token：`{payload['token_usage']}` / `{payload['token_cap']}`",
        f"- Stable 自动修改：`{str(payload['stable_skill_mutation_enabled']).lower()}`",
        f"- 生产 Candidate 加载：`{str(payload['production_loading_enabled']).lower()}`",
        "",
    ]
    if not payload["replays"]:
        if candidate["status"] == "preflight_only":
            lines.extend(
                [
                    "## Stable 预检",
                    "",
                    "| Fixture | 指标 | 失败数 | Headroom |",
                    "|---|---|---:|---|",
                ]
            )
            for item in payload["stable_preflight"]:
                lines.append(
                    f"| {item['fixture_id']} | {item['target_metric']} | "
                    f"{item['metric_value']:.0f} | "
                    f"{str(item['headroom_available']).lower()} |"
                )
            lines.append("")
        else:
            result_text = (
                "Stable 没有可测改进空间，已在生成前跳过 Candidate。"
                if candidate["status"] == "skipped_no_headroom"
                else "Candidate 未获准进入 replay。"
            )
            lines.extend(["## 结果", "", result_text, ""])
    else:
        metric_label = payload.get("target_metric", "target_metric")
        lines.extend(
            [
                f"| Fixture | 状态 | Oracle | Stable {metric_label} | Candidate {metric_label} | VerifierGap |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for item in payload["replays"]:
            lines.append(
                f"| {item['fixture_id']} | {item['status']} | {item['oracle_decision']} | "
                f"{item['stable_metric']:.0f} | "
                f"{item['candidate_metric']:.0f} | "
                f"{item['verifier_gap_id'] or '-'} |"
            )
        lines.append("")
    lines.extend(["## 准入结论", "", *candidate["final_reasons"], ""])
    return "\n".join(lines)


async def _execute_query_selection(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "queries": fixture.input_snapshot["queries"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_QUERY_SELECTOR_SYSTEM,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        _QUERY_SELECTOR_SYSTEM,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)
    raw_ids = value.get("selected_query_ids", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    valid_ids = {
        str(item["query_id"]) for item in fixture.input_snapshot["queries"]
    }
    groups = fixture.input_snapshot["equivalence_groups"]
    counts = [sum(item in selected for item in group) for group in groups]
    invalid_ids = [item for item in selected if item not in valid_ids]
    duplicate_queries = sum(max(0, count - 1) for count in counts)
    hard_checks = {
        "valid_query_ids_only": not invalid_ids,
        "all_intents_preserved": all(count >= 1 for count in counts),
        "one_query_per_intent": all(count == 1 for count in counts),
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={"duplicate_queries": float(duplicate_queries)},
        hard_checks=hard_checks,
        output={
            "selected_query_ids": selected,
            "rationale": str(value.get("rationale") or ""),
            "invalid_query_ids": invalid_ids,
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_route_selection(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "prior_failed_source_families": fixture.input_snapshot[
            "prior_failed_source_families"
        ],
        # Simulated outcomes belong to the deterministic Oracle and are hidden
        # from both arms to prevent answer leakage.
        "available_routes": [
            {key: value for key, value in item.items() if key != "outcome"}
            for item in fixture.input_snapshot["available_routes"]
        ],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_ROUTE_SELECTOR_SYSTEM,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        _ROUTE_SELECTOR_SYSTEM,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)
    raw_ids = value.get("selected_route_ids", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    routes = {
        str(item["route_id"]): item
        for item in fixture.input_snapshot["available_routes"]
    }
    invalid_ids = [item for item in selected if item not in routes]
    blocked_ids: list[str] = []
    attempted_ids: list[str] = []
    source_family_failures = 0
    success_reached = False
    exhausted = set(fixture.input_snapshot["prior_failed_source_families"])
    for route_id in selected:
        route = routes.get(route_id)
        if route is None:
            continue
        attempted_ids.append(route_id)
        if not bool(route.get("allowed", False)):
            blocked_ids.append(route_id)
            continue
        if str(route.get("source_family") or "") in exhausted:
            source_family_failures += 1
        if route.get("outcome") == "success":
            success_reached = True
            break
    hard_checks = {
        "valid_route_ids_only": not invalid_ids,
        "blocked_routes_avoided": not blocked_ids,
        "successful_route_reached": success_reached,
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={"source_family_failures": float(source_family_failures)},
        hard_checks=hard_checks,
        output={
            "selected_route_ids": selected,
            "attempted_route_ids": attempted_ids,
            "invalid_route_ids": invalid_ids,
            "blocked_route_ids": blocked_ids,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_route_selection_history(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    snapshot = fixture.input_snapshot
    user_payload = {
        "skill_instructions": instructions,
        "prior_attempts": snapshot["prior_attempts"],
        # Outcomes and normalized family aliases are Oracle-only data.
        "available_routes": [
            {key: value for key, value in item.items() if key != "outcome"}
            for item in snapshot["available_routes"]
        ],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_ROUTE_HISTORY_SELECTOR_SYSTEM,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        _ROUTE_HISTORY_SELECTOR_SYSTEM,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)
    raw_ids = value.get("selected_route_ids", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    routes = {
        str(item["route_id"]): item for item in snapshot["available_routes"]
    }
    exhausted = {
        str(item) for item in snapshot["oracle_exhausted_source_families"]
    }
    invalid_ids = [item for item in selected if item not in routes]
    blocked_ids: list[str] = []
    attempted_ids: list[str] = []
    source_family_failures = 0
    success_reached = False
    for route_id in selected:
        route = routes.get(route_id)
        if route is None:
            continue
        attempted_ids.append(route_id)
        if not bool(route.get("allowed", False)):
            blocked_ids.append(route_id)
            continue
        if str(route.get("source_family") or "") in exhausted:
            source_family_failures += 1
        if route.get("outcome") == "success":
            success_reached = True
            break
    hard_checks = {
        "valid_route_ids_only": not invalid_ids,
        "blocked_routes_avoided": not blocked_ids,
        "successful_route_reached": success_reached,
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={"source_family_failures": float(source_family_failures)},
        hard_checks=hard_checks,
        output={
            "selected_route_ids": selected,
            "attempted_route_ids": attempted_ids,
            "invalid_route_ids": invalid_ids,
            "blocked_route_ids": blocked_ids,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_source_selection(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "prior_selected_domains": fixture.input_snapshot[
            "prior_selected_domains"
        ],
        "selection_count": fixture.input_snapshot["selection_count"],
        "candidate_sources": fixture.input_snapshot["candidate_sources"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_SOURCE_SELECTOR_SYSTEM,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        _SOURCE_SELECTOR_SYSTEM,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)
    raw_ids = value.get("selected_source_ids", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    sources = {
        str(item["source_id"]): item
        for item in fixture.input_snapshot["candidate_sources"]
    }
    invalid_ids = [item for item in selected if item not in sources]
    selected_sources = [sources[item] for item in selected if item in sources]
    blocked_ids = [
        str(item["source_id"])
        for item in selected_sources
        if not bool(item.get("allowed", False))
    ]
    prior_domains = set(fixture.input_snapshot["prior_selected_domains"])
    repeated = sum(
        str(item.get("domain") or "") in prior_domains
        for item in selected_sources
    )
    repeated_rate = repeated / len(selected_sources) if selected_sources else 1.0
    authoritative = {"primary", "authoritative", "authoritative-copy"}
    hard_checks = {
        "valid_source_ids_only": not invalid_ids,
        "selection_count_met": len(selected) == fixture.input_snapshot[
            "selection_count"
        ],
        "blocked_sources_avoided": not blocked_ids,
        "authoritative_source_preserved": any(
            str(item.get("quality") or "") in authoritative
            for item in selected_sources
        ),
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={"repeated_domain_rate": float(repeated_rate)},
        hard_checks=hard_checks,
        output={
            "selected_source_ids": selected,
            "invalid_source_ids": invalid_ids,
            "blocked_source_ids": blocked_ids,
            "selected_domains": [item.get("domain") for item in selected_sources],
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_query_reformulation(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "research_scope": fixture.input_snapshot["research_scope"],
        "prior_queries": fixture.input_snapshot["prior_queries"],
        "candidate_queries": [
            {
                key: value
                for key, value in item.items()
                if key not in {"outcome", "in_scope", "dimension", "strategy"}
            }
            for item in fixture.input_snapshot["candidate_queries"]
        ],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_REFORMULATION_SYSTEM,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        _REFORMULATION_SYSTEM,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)
    raw_ids = value.get("selected_query_ids", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    queries = {
        str(item["query_id"]): item
        for item in fixture.input_snapshot["candidate_queries"]
    }
    invalid_ids = [item for item in selected if item not in queries]
    selected_queries = [queries[item] for item in selected if item in queries]
    evidence_gain = any(item.get("outcome") == "gain" for item in selected_queries)
    scope_preserved = bool(selected_queries) and all(
        bool(item.get("in_scope", False)) for item in selected_queries
    )
    zero_gain_searches = sum(
        item.get("outcome") != "gain" for item in selected_queries
    )
    hard_checks = {
        "valid_query_ids_only": not invalid_ids,
        "selection_count_met": len(selected) == 1,
        "research_scope_preserved": scope_preserved,
        "evidence_gain_reached": evidence_gain,
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={
            "zero_gain_searches": float(zero_gain_searches),
            "evidence_gain": 1.0 if evidence_gain else 0.0,
        },
        hard_checks=hard_checks,
        output={
            "selected_query_ids": selected,
            "invalid_query_ids": invalid_ids,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_entity_materialization(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "research_scope": fixture.input_snapshot["research_scope"],
        "source_documents": fixture.input_snapshot["source_documents"],
        "candidate_entities": fixture.input_snapshot["candidate_entities"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_ENTITY_MATERIALIZATION_SYSTEM,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        _ENTITY_MATERIALIZATION_SYSTEM,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)
    raw_ids = value.get("selected_entity_ids", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    valid_ids = {
        str(item["entity_id"])
        for item in fixture.input_snapshot["candidate_entities"]
    }
    gold_ids = set(fixture.input_snapshot["gold_entity_ids"])
    selected_set = set(selected)
    invalid_ids = [item for item in selected if item not in valid_ids]
    false_positive_ids = sorted((selected_set & valid_ids) - gold_ids)
    missing_entity_ids = sorted(gold_ids - selected_set)
    entity_errors = len(invalid_ids) + len(false_positive_ids) + len(
        missing_entity_ids
    )
    container_ids = set(fixture.input_snapshot.get("container_entity_ids", []))
    required_documents = set(
        fixture.input_snapshot.get("required_document_entity_ids", [])
    )
    hard_checks = {
        "valid_entity_ids_only": not invalid_ids,
        "target_entity_set_complete": not false_positive_ids
        and not missing_entity_ids,
        "container_entities_excluded": not (selected_set & container_ids),
        "target_documents_preserved": required_documents <= selected_set,
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={"entity_errors": float(entity_errors)},
        hard_checks=hard_checks,
        output={
            "selected_entity_ids": selected,
            "invalid_entity_ids": invalid_ids,
            "false_positive_entity_ids": false_positive_ids,
            "missing_entity_ids": missing_entity_ids,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _normalize_entity_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _entity_name_sets_overlap(predicted: set[str], gold: set[str]) -> bool:
    """Allow harmless title qualifiers without fuzzy-matching short aliases."""

    for predicted_name in predicted:
        for gold_name in gold:
            if predicted_name == gold_name:
                return True
            if min(len(predicted_name), len(gold_name)) >= 8 and (
                predicted_name in gold_name or gold_name in predicted_name
            ):
                return True
    return False


async def _execute_entity_discovery(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "research_scope": fixture.input_snapshot["research_scope"],
        "source_documents": fixture.input_snapshot["source_documents"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    system_prompt = (
        _ENTITY_DISCOVERY_NEUTRAL_SYSTEM
        if fixture.input_snapshot.get("neutral_system")
        else _ENTITY_DISCOVERY_SYSTEM
    )
    budget.preflight(
        system=system_prompt,
        user=user,
        max_tokens=max_tokens,
    )
    value, response = await llm.complete_json(
        system_prompt,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)

    raw_entities = value.get("entities", [])
    predicted: list[dict[str, Any]] = []
    if isinstance(raw_entities, list):
        for item in raw_entities:
            if isinstance(item, str):
                name = item.strip()
                aliases: list[str] = []
                source_ids: list[str] = []
            elif isinstance(item, dict):
                name = str(
                    item.get("canonical_name")
                    or item.get("name")
                    or item.get("label")
                    or ""
                ).strip()
                raw_aliases = item.get("aliases", [])
                aliases = (
                    [str(alias).strip() for alias in raw_aliases if str(alias).strip()]
                    if isinstance(raw_aliases, list)
                    else []
                )
                raw_source_ids = item.get("source_ids", [])
                source_ids = (
                    [
                        str(source_id).strip()
                        for source_id in raw_source_ids
                        if str(source_id).strip()
                    ]
                    if isinstance(raw_source_ids, list)
                    else []
                )
            else:
                continue
            if name:
                predicted.append(
                    {
                        "canonical_name": name,
                        "aliases": aliases,
                        "source_ids": source_ids,
                    }
                )

    gold_aliases: dict[str, set[str]] = {}
    for item in fixture.input_snapshot["gold_entities"]:
        entity_id = str(item["entity_id"])
        names = [item["canonical_name"], *item.get("aliases", [])]
        gold_aliases[entity_id] = {
            normalized
            for name in names
            if (normalized := _normalize_entity_name(name))
        }

    matched_ids: set[str] = set()
    unsupported_names: list[str] = []
    duplicate_or_merged_names: list[str] = []
    container_names = {
        _normalize_entity_name(name)
        for name in fixture.input_snapshot.get("container_names", [])
    }
    selected_container_names: list[str] = []
    source_binding_error_names: list[str] = []
    source_bound_ids: set[str] = set()
    valid_source_ids = {
        str(item["source_id"])
        for item in fixture.input_snapshot["source_documents"]
    }
    expected_sources_by_entity = {
        str(item["entity_id"]): {
            str(source_id) for source_id in item.get("source_ids", [])
        }
        for item in fixture.input_snapshot["gold_entities"]
        if item.get("source_ids")
    }
    for item in predicted:
        prediction_names = {
            normalized
            for name in [item["canonical_name"], *item["aliases"]]
            if (normalized := _normalize_entity_name(name))
        }
        if prediction_names & container_names:
            selected_container_names.append(item["canonical_name"])
        matching_gold = [
            entity_id
            for entity_id, aliases in gold_aliases.items()
            if _entity_name_sets_overlap(prediction_names, aliases)
        ]
        if len(matching_gold) != 1:
            unsupported_names.append(item["canonical_name"])
            if len(matching_gold) > 1:
                duplicate_or_merged_names.append(item["canonical_name"])
            continue
        entity_id = matching_gold[0]
        if entity_id in matched_ids:
            duplicate_or_merged_names.append(item["canonical_name"])
            continue
        matched_ids.add(entity_id)
        expected_sources = expected_sources_by_entity.get(entity_id, set())
        if expected_sources:
            predicted_sources = set(item["source_ids"])
            if (
                not predicted_sources
                or not predicted_sources <= valid_source_ids
                or not (predicted_sources & expected_sources)
            ):
                source_binding_error_names.append(item["canonical_name"])
            else:
                source_bound_ids.add(entity_id)

    missing_ids = sorted(set(gold_aliases) - matched_ids)
    predicted_completeness = str(
        value.get("scope_completeness_status") or ""
    ).strip().casefold()
    expected_completeness = {
        str(item).strip().casefold()
        for item in fixture.input_snapshot.get(
            "expected_completeness_statuses", []
        )
        if str(item).strip()
    }
    completeness_ok = (
        not expected_completeness
        or predicted_completeness in expected_completeness
    )
    failures = (
        len(missing_ids)
        + len(unsupported_names)
        + len(duplicate_or_merged_names)
        + len(source_binding_error_names)
        + (0 if completeness_ok else 1)
    )
    gold_count = len(gold_aliases)
    prediction_count = len(predicted)
    hard_checks = {
        "all_target_entities_found": not missing_ids,
        "no_unsupported_entities": not unsupported_names,
        "aliases_not_split_or_merged": not duplicate_or_merged_names,
        "container_entities_excluded": not selected_container_names,
        "entity_sources_valid": not source_binding_error_names,
        "scope_completeness_calibrated": completeness_ok,
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={
            "skill_execution_failures": float(failures),
            "entity_recall": len(matched_ids) / gold_count if gold_count else 1.0,
            "entity_precision": (
                len(matched_ids) / prediction_count if prediction_count else 0.0
            ),
            "entity_source_binding_accuracy": (
                len(source_bound_ids) / len(expected_sources_by_entity)
                if expected_sources_by_entity
                else 1.0
            ),
        },
        hard_checks=hard_checks,
        output={
            "entities": predicted,
            "matched_entity_ids": sorted(matched_ids),
            "missing_entity_ids": missing_ids,
            "unsupported_entity_names": unsupported_names,
            "duplicate_or_merged_entity_names": duplicate_or_merged_names,
            "selected_container_names": selected_container_names,
            "source_binding_error_names": source_binding_error_names,
            "scope_completeness_status": predicted_completeness,
            "expected_completeness_statuses": sorted(expected_completeness),
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_field_matrix(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    user_payload = {
        "skill_instructions": instructions,
        "research_scope": fixture.input_snapshot["research_scope"],
        "entities": fixture.input_snapshot["entities"],
        "field_schema": fixture.input_snapshot["field_schema"],
        "source_documents": fixture.input_snapshot["source_documents"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    system_prompt = (
        _FIELD_MATRIX_NEUTRAL_SYSTEM
        if fixture.input_snapshot.get("neutral_system")
        else _FIELD_MATRIX_SYSTEM
    )
    budget.preflight(system=system_prompt, user=user, max_tokens=max_tokens)
    value, response = await llm.complete_json(
        system_prompt,
        user,
        max_tokens=max_tokens,
    )
    budget.settle(response)

    source_text = {
        str(item["source_id"]): str(item["text"])
        for item in fixture.input_snapshot["source_documents"]
    }
    valid_entities = {
        str(item["entity_id"]) for item in fixture.input_snapshot["entities"]
    }
    valid_fields = {
        str(item["field_key"]) for item in fixture.input_snapshot["field_schema"]
    }
    gold_by_key = {
        (str(item["entity_id"]), str(item["field_key"])): item
        for item in fixture.input_snapshot["gold_cells"]
    }

    raw_cells = value.get("cells", [])
    parsed_cells: list[dict[str, Any]] = []
    if isinstance(raw_cells, list):
        for item in raw_cells:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id") or "")
            field_key = str(item.get("field_key") or "")
            status = str(item.get("status") or "").casefold()
            raw_values = item.get("values", [])
            parsed_values: list[dict[str, str]] = []
            if isinstance(raw_values, list):
                for raw_value in raw_values:
                    if not isinstance(raw_value, dict):
                        continue
                    parsed_values.append(
                        {
                            "value": str(raw_value.get("value") or "").strip(),
                            "source_id": str(
                                raw_value.get("source_id") or ""
                            ).strip(),
                            "quote": str(raw_value.get("quote") or "").strip(),
                        }
                    )
            parsed_cells.append(
                {
                    "entity_id": entity_id,
                    "field_key": field_key,
                    "status": status,
                    "values": parsed_values,
                }
            )

    predicted_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    extra_keys: list[str] = []
    invalid_binding_keys: list[str] = []
    for cell in parsed_cells:
        key = (cell["entity_id"], cell["field_key"])
        key_label = f"{key[0]}::{key[1]}"
        if key[0] not in valid_entities or key[1] not in valid_fields:
            invalid_binding_keys.append(key_label)
        if key not in gold_by_key:
            extra_keys.append(key_label)
        if key in predicted_by_key:
            duplicate_keys.append(key_label)
        else:
            predicted_by_key[key] = cell

    missing_keys: list[str] = []
    wrong_status_keys: list[str] = []
    wrong_value_keys: list[str] = []
    invalid_evidence_keys: list[str] = []
    invalid_evidence_details: list[dict[str, str]] = []
    failed_gold_keys: set[tuple[str, str]] = set()
    nr_total = 0
    nr_correct = 0
    for key, gold in gold_by_key.items():
        key_label = f"{key[0]}::{key[1]}"
        predicted = predicted_by_key.get(key)
        if predicted is None:
            missing_keys.append(key_label)
            failed_gold_keys.add(key)
            continue
        expected_status = str(gold["status"]).casefold()
        if expected_status == "nr":
            nr_total += 1
        status_ok = predicted["status"] == expected_status
        if not status_ok:
            wrong_status_keys.append(key_label)

        predicted_values = {
            _normalize_entity_name(item["value"]) for item in predicted["values"]
        }
        expected_values = {
            _normalize_entity_name(item["value"])
            for item in gold.get("expected_values", [])
        }
        values_ok = predicted_values == expected_values
        if not values_ok:
            wrong_value_keys.append(key_label)

        evidence_ok = True
        expected_sources_by_value = {
            _normalize_entity_name(item["value"]): set(item["source_ids"])
            for item in gold.get("expected_values", [])
        }
        required_quote_terms_by_value = {
            _normalize_entity_name(item["value"]): [
                _normalize_entity_name(term)
                for term in item.get("quote_must_contain_any", [])
                if _normalize_entity_name(term)
            ]
            for item in gold.get("expected_values", [])
        }
        for item in predicted["values"]:
            normalized_value = _normalize_entity_name(item["value"])
            source_id = item["source_id"]
            quote = item["quote"]
            normalized_quote = _normalize_entity_name(quote)
            reason = ""
            if source_id not in source_text:
                reason = "unknown_source_id"
            elif not quote:
                reason = "missing_quote"
            elif quote not in source_text[source_id]:
                reason = "quote_is_not_an_exact_contiguous_source_substring"
            elif source_id not in expected_sources_by_value.get(
                normalized_value, set()
            ):
                reason = "source_does_not_support_expected_value"
            elif required_quote_terms_by_value.get(normalized_value) and not any(
                term in normalized_quote
                for term in required_quote_terms_by_value[normalized_value]
            ):
                reason = "quote_does_not_identify_the_target_entity"
            if reason:
                evidence_ok = False
                invalid_evidence_details.append(
                    {
                        "cell_key": key_label,
                        "value": item["value"],
                        "source_id": source_id,
                        "quote": quote,
                        "reason": reason,
                        "source_excerpt": source_text.get(source_id, "")[:600],
                    }
                )
                break
        if expected_status == "nr" and predicted["values"]:
            evidence_ok = False
        if not evidence_ok:
            invalid_evidence_keys.append(key_label)

        if status_ok and values_ok and evidence_ok:
            if expected_status == "nr":
                nr_correct += 1
        else:
            failed_gold_keys.add(key)

    failures = (
        len(failed_gold_keys)
        + len(extra_keys)
        + len(duplicate_keys)
        + len(invalid_binding_keys)
    )
    gold_count = len(gold_by_key)
    hard_checks = {
        "all_cells_present": not missing_keys,
        "no_extra_or_duplicate_cells": not extra_keys and not duplicate_keys,
        "all_bindings_valid": not invalid_binding_keys,
        "statuses_correct": not wrong_status_keys,
        "values_correct": not wrong_value_keys,
        "quotes_exact_and_sources_valid": not invalid_evidence_keys,
        "nr_not_fabricated": nr_correct == nr_total,
    }
    return LocalReplayArmResult(
        arm=arm,
        metrics={
            "skill_execution_failures": float(failures),
            "cell_accuracy": (
                (gold_count - len(failed_gold_keys)) / gold_count
                if gold_count
                else 1.0
            ),
            "cell_coverage": (
                (gold_count - len(missing_keys)) / gold_count
                if gold_count
                else 1.0
            ),
            "nr_correctness": nr_correct / nr_total if nr_total else 1.0,
        },
        hard_checks=hard_checks,
        output={
            "cells": parsed_cells,
            "missing_cell_keys": missing_keys,
            "extra_cell_keys": extra_keys,
            "duplicate_cell_keys": duplicate_keys,
            "invalid_binding_keys": invalid_binding_keys,
            "wrong_status_keys": wrong_status_keys,
            "wrong_value_keys": wrong_value_keys,
            "invalid_evidence_keys": invalid_evidence_keys,
            "invalid_evidence_details": invalid_evidence_details,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_task_contract_decomposition(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    snapshot = fixture.input_snapshot
    user_payload = {
        "skill_instructions": instructions,
        "overall_research_scope": snapshot["research_scope"],
        "current_coverage": snapshot["current_coverage"],
        "available_source": snapshot["available_source"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(system=_TASK_CONTRACT_SYSTEM, user=user, max_tokens=max_tokens)
    value, response = await llm.complete_json(
        _TASK_CONTRACT_SYSTEM, user, max_tokens=max_tokens
    )
    budget.settle(response)
    raw_contract = value.get("task_contract")
    contract = raw_contract if isinstance(raw_contract, dict) else {}

    def _strings(key: str) -> list[str]:
        raw = contract.get(key, [])
        return (
            list(dict.fromkeys(str(item) for item in raw if str(item)))
            if isinstance(raw, list)
            else []
        )

    source_ids = _strings("source_ids")
    target_entity_ids = _strings("target_entity_ids")
    target_field_keys = _strings("target_field_keys")
    queries = _strings("queries")
    expected_source_ids = {str(snapshot["available_source"]["source_id"])}
    supported_fields = {
        str(item) for item in snapshot["available_source"]["supported_field_keys"]
    }
    required_fields = {
        str(item) for item in snapshot["oracle"]["required_field_keys"]
    }
    allowed_contracts = {
        str(item) for item in snapshot["oracle"]["allowed_completion_contracts"]
    }
    hard_checks = {
        "task_contract_is_structured": bool(contract),
        "bounded_to_available_source": set(source_ids) == expected_source_ids,
        "no_redundant_search": not queries,
        "partial_results_allowed": contract.get("allow_partial_results") is True,
        "unresolved_fields_allowed": contract.get("accept_unresolved_fields") is True,
        "local_completion_contract": str(contract.get("completion_contract") or "")
        in allowed_contracts,
        "required_local_fields_covered": required_fields <= set(target_field_keys),
        "unsupported_fields_not_requested": set(target_field_keys) <= supported_fields,
        "no_premature_entity_id_invention": not target_entity_ids,
        "objective_present": bool(str(contract.get("objective") or "").strip()),
    }
    failures = sum(not value for value in hard_checks.values())
    return LocalReplayArmResult(
        arm=arm,
        metrics={"skill_execution_failures": float(failures)},
        hard_checks=hard_checks,
        output={
            "task_contract": contract,
            "failed_checks": [
                name for name, passed in hard_checks.items() if not passed
            ],
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_coverage_gap_planning(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    snapshot = fixture.input_snapshot
    user_payload = {
        "skill_instructions": instructions,
        "research_scope": snapshot["research_scope"],
        "required_cells": snapshot["required_cells"],
        "current_coverage": snapshot["current_coverage"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(
        system=_COVERAGE_GAP_PLANNING_SYSTEM, user=user, max_tokens=max_tokens
    )
    value, response = await llm.complete_json(
        _COVERAGE_GAP_PLANNING_SYSTEM, user, max_tokens=max_tokens
    )
    budget.settle(response)

    raw_keys = value.get("selected_cell_keys", [])
    selected = (
        list(dict.fromkeys(str(item) for item in raw_keys if str(item)))
        if isinstance(raw_keys, list)
        else []
    )
    valid_keys = {str(item["cell_key"]) for item in snapshot["required_cells"]}
    expected_keys = {
        str(item) for item in snapshot["oracle"]["expected_target_cell_keys"]
    }
    forbidden_keys = {
        str(item) for item in snapshot["oracle"].get("forbidden_target_cell_keys", [])
    }
    selected_set = set(selected)
    invalid_keys = sorted(selected_set - valid_keys)
    missing_keys = sorted(expected_keys - selected_set)
    redundant_keys = sorted(selected_set & forbidden_keys)
    unexpected_keys = sorted((selected_set & valid_keys) - expected_keys)
    predicted_status = str(value.get("scope_completeness_status") or "").casefold()
    allowed_statuses = {
        str(item).casefold()
        for item in snapshot["oracle"]["allowed_scope_completeness_statuses"]
    }
    hard_checks = {
        "valid_cell_keys_only": not invalid_keys,
        "all_material_gaps_targeted": not missing_keys,
        "supported_cells_not_researched": not redundant_keys,
        "no_unexpected_targets": not unexpected_keys,
        "scope_completeness_calibrated": predicted_status in allowed_statuses,
    }
    failures = sum(not passed for passed in hard_checks.values())
    return LocalReplayArmResult(
        arm=arm,
        metrics={"skill_execution_failures": float(failures)},
        hard_checks=hard_checks,
        output={
            "selected_cell_keys": selected,
            "invalid_cell_keys": invalid_keys,
            "missing_cell_keys": missing_keys,
            "redundant_cell_keys": redundant_keys,
            "unexpected_cell_keys": unexpected_keys,
            "scope_completeness_status": predicted_status,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def _execute_analysis_chain(
    *,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    snapshot = fixture.input_snapshot
    user_payload = {
        "skill_instructions": instructions,
        "research_scope": snapshot["research_scope"],
        "evidence_items": snapshot["evidence_items"],
        "analysis_requests": snapshot["analysis_requests"],
    }
    user = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
    budget.preflight(system=_ANALYSIS_CHAIN_SYSTEM, user=user, max_tokens=max_tokens)
    value, response = await llm.complete_json(
        _ANALYSIS_CHAIN_SYSTEM, user, max_tokens=max_tokens
    )
    budget.settle(response)

    raw_units = value.get("analysis_units", [])
    units = [item for item in raw_units if isinstance(item, dict)] if isinstance(raw_units, list) else []
    valid_request_ids = {
        str(item["request_id"]) for item in snapshot["analysis_requests"]
    }
    valid_evidence_ids = {
        str(item["evidence_id"]) for item in snapshot["evidence_items"]
    }
    units_by_request: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        units_by_request.setdefault(str(unit.get("request_id") or ""), []).append(unit)

    missing_requests: list[str] = []
    duplicate_requests: list[str] = []
    invalid_request_ids = sorted(set(units_by_request) - valid_request_ids)
    invalid_evidence_requests: list[str] = []
    missing_premise_requests: list[str] = []
    missing_concept_requests: list[str] = []
    for rule in snapshot["oracle"]["request_rules"]:
        request_id = str(rule["request_id"])
        matching = units_by_request.get(request_id, [])
        if not matching:
            missing_requests.append(request_id)
            continue
        if len(matching) != 1:
            duplicate_requests.append(request_id)
        unit = matching[0]
        raw_refs = unit.get("premise_evidence_ids", [])
        refs = {str(item) for item in raw_refs} if isinstance(raw_refs, list) else set()
        if not refs <= valid_evidence_ids:
            invalid_evidence_requests.append(request_id)
        required_refs = {str(item) for item in rule.get("required_evidence_ids", [])}
        if not required_refs <= refs:
            missing_premise_requests.append(request_id)
        analysis_text = _normalize_entity_name(
            f"{unit.get('reasoning') or ''} {unit.get('conclusion') or ''}"
        )
        concept_groups = rule.get("must_contain_any_groups", [])
        if any(
            not any(_normalize_entity_name(term) in analysis_text for term in group)
            for group in concept_groups
        ):
            missing_concept_requests.append(request_id)

    hard_checks = {
        "all_analysis_requests_answered": not missing_requests,
        "one_unit_per_request": not duplicate_requests and not invalid_request_ids,
        "valid_evidence_ids_only": not invalid_evidence_requests,
        "required_premises_used": not missing_premise_requests,
        "reasoning_concepts_preserved": not missing_concept_requests,
    }
    failures = sum(not passed for passed in hard_checks.values())
    return LocalReplayArmResult(
        arm=arm,
        metrics={"skill_execution_failures": float(failures)},
        hard_checks=hard_checks,
        output={
            "analysis_units": units,
            "missing_request_ids": missing_requests,
            "duplicate_request_ids": duplicate_requests,
            "invalid_request_ids": invalid_request_ids,
            "invalid_evidence_request_ids": invalid_evidence_requests,
            "missing_premise_request_ids": missing_premise_requests,
            "missing_concept_request_ids": missing_concept_requests,
            "rationale": str(value.get("rationale") or ""),
        },
        model_calls=1,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
def _fixture_from_raw(
    raw: dict[str, Any],
    *,
    episode: EvolutionEpisode,
    scenario_type: str,
    fixture_id: str,
    target_metric: str,
    invariants: list[str],
    arm_token_cap: int,
) -> LocalMicroFixture:
    if scenario_type == "query_selection":
        snapshot = {
            "queries": raw["queries"],
            "equivalence_groups": raw["equivalence_groups"],
        }
    elif scenario_type == "route_selection":
        snapshot = {
            "prior_failed_source_families": raw[
                "prior_failed_source_families"
            ],
            "available_routes": raw["available_routes"],
        }
    elif scenario_type == "route_selection_history":
        snapshot = {
            "prior_attempts": raw["prior_attempts"],
            "available_routes": raw["available_routes"],
            # Normalized family aliases are deterministic Oracle data and are
            # never included in the model payload.
            "oracle_exhausted_source_families": raw[
                "oracle_exhausted_source_families"
            ],
        }
    elif scenario_type == "source_selection":
        snapshot = {
            "prior_selected_domains": raw["prior_selected_domains"],
            "selection_count": raw["selection_count"],
            "candidate_sources": raw["candidate_sources"],
        }
    elif scenario_type == "query_reformulation":
        snapshot = {
            "research_scope": raw["research_scope"],
            "prior_queries": raw["prior_queries"],
            "candidate_queries": raw["candidate_queries"],
        }
    elif scenario_type == "entity_materialization":
        snapshot = {
            "research_scope": raw["research_scope"],
            "source_documents": raw["source_documents"],
            "candidate_entities": raw["candidate_entities"],
            # These gold fields remain inside the deterministic Oracle. The
            # executor deliberately omits them from the model payload.
            "gold_entity_ids": raw["gold_entity_ids"],
            "container_entity_ids": raw.get("container_entity_ids", []),
            "required_document_entity_ids": raw.get(
                "required_document_entity_ids", []
            ),
        }
    elif scenario_type in {"entity_discovery", "entity_discovery_neutral"}:
        snapshot = {
            "research_scope": raw["research_scope"],
            "source_documents": raw["source_documents"],
            # Gold aliases and container names are deterministic Oracle data
            # and are never included in the model payload.
            "gold_entities": raw["gold_entities"],
            "container_names": raw.get("container_names", []),
            "expected_completeness_statuses": raw.get(
                "expected_completeness_statuses", []
            ),
            "neutral_system": scenario_type == "entity_discovery_neutral",
        }
    elif scenario_type in {"field_matrix", "field_matrix_neutral"}:
        snapshot = {
            "research_scope": raw["research_scope"],
            "entities": raw["entities"],
            "field_schema": raw["field_schema"],
            "source_documents": raw["source_documents"],
            # Gold cells remain available only to the deterministic Oracle.
            "gold_cells": raw["gold_cells"],
            "neutral_system": scenario_type == "field_matrix_neutral",
        }
    elif scenario_type == "task_contract_decomposition":
        snapshot = {
            "research_scope": raw["research_scope"],
            "current_coverage": raw["current_coverage"],
            "available_source": raw["available_source"],
            # Oracle policy is not included in the model payload.
            "oracle": raw["oracle"],
        }
    elif scenario_type == "coverage_gap_planning":
        snapshot = {
            "research_scope": raw["research_scope"],
            "required_cells": raw["required_cells"],
            "current_coverage": raw["current_coverage"],
            # Expected targets are deterministic Oracle data and are hidden.
            "oracle": raw["oracle"],
        }
    elif scenario_type == "analysis_chain":
        snapshot = {
            "research_scope": raw["research_scope"],
            "evidence_items": raw["evidence_items"],
            "analysis_requests": raw["analysis_requests"],
            # Premise/concept rules are deterministic Oracle data and hidden.
            "oracle": raw["oracle"],
        }
    else:
        raise ValueError(f"Unsupported scenario_type: {scenario_type}")
    return LocalMicroFixture(
        fixture_id=fixture_id,
        target_skill_id=episode.target_skill_id,
        trigger_type=episode.trigger_types[0],
        input_snapshot=snapshot,
        target_metric=target_metric,
        fixture_role=str(raw.get("fixture_role") or "target"),
        invariants=invariants,
        max_model_calls=1,
        max_tokens=arm_token_cap,
    )


async def _execute_fixture(
    *,
    scenario_type: str,
    fixture: LocalMicroFixture,
    instructions: str,
    arm: str,
    llm,
    budget: CampaignBudget,
    max_tokens: int,
) -> LocalReplayArmResult:
    executor = {
        "query_selection": _execute_query_selection,
        "route_selection": _execute_route_selection,
        "route_selection_history": _execute_route_selection_history,
        "source_selection": _execute_source_selection,
        "query_reformulation": _execute_query_reformulation,
        "entity_materialization": _execute_entity_materialization,
        "entity_discovery": _execute_entity_discovery,
        "entity_discovery_neutral": _execute_entity_discovery,
        "field_matrix": _execute_field_matrix,
        "field_matrix_neutral": _execute_field_matrix,
        "task_contract_decomposition": _execute_task_contract_decomposition,
        "coverage_gap_planning": _execute_coverage_gap_planning,
        "analysis_chain": _execute_analysis_chain,
    }.get(scenario_type)
    if executor is None:
        raise ValueError(f"Unsupported scenario_type: {scenario_type}")
    return await executor(
        fixture=fixture,
        instructions=instructions,
        arm=arm,
        llm=llm,
        budget=budget,
        max_tokens=max_tokens,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    campaign_path = _resolve(args.campaign)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_id = str(campaign["campaign_id"])
    scenario_type = str(campaign.get("scenario_type") or "query_selection")
    target_metric = str(campaign.get("target_metric") or "duplicate_queries")
    invariants = [str(item) for item in campaign.get("invariants", [])]
    if not invariants and scenario_type == "query_selection":
        invariants = [
            "valid_query_ids_only",
            "all_intents_preserved",
            "one_query_per_intent",
        ]
    raw_episode = dict(campaign["episode"])
    if args.attempt_id:
        if not args.attempt_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("attempt_id contains unsafe characters")
        campaign_id = f"{campaign_id}-{args.attempt_id}"
        raw_episode["episode_id"] = (
            f"{raw_episode['episode_id']}-{args.attempt_id}"
        )
        raw_episode["run_id"] = f"{raw_episode['run_id']}-{args.attempt_id}"
        raw_episode["trigger_signal_ids"] = [
            f"{item}-{args.attempt_id}"
            for item in raw_episode["trigger_signal_ids"]
        ]
    runtime = load_runtime_config(_resolve(args.config))
    router = ModelRouter(runtime)
    registry = load_skill_registry(ROOT)
    store_root = _resolve(args.store_root)
    artifact_root = _resolve(args.artifact_root)
    output_dir = _resolve(args.output_dir)
    campaign_episode = EvolutionEpisode.model_validate(raw_episode)
    frozen_candidate_path = (
        _resolve(args.candidate_file) if args.candidate_file else None
    )
    frozen_candidate: CandidateRecord | None = None
    if frozen_candidate_path is not None:
        frozen_candidate = CandidateRecord.model_validate(
            json.loads(frozen_candidate_path.read_text(encoding="utf-8"))
        )
        if (
            frozen_candidate.status != "accept_for_replay"
            or frozen_candidate.final_patch is None
        ):
            raise ValueError(
                "--candidate-file must contain an accept_for_replay Candidate"
            )
        if frozen_candidate.target_skill_id != campaign_episode.target_skill_id:
            raise ValueError("Frozen Candidate targets a different Skill")
        source_store_root = frozen_candidate_path.parent.parent
        episode_store = EvolutionEpisodeStore(source_store_root)
        candidate_store = CandidateStore(source_store_root)
        episode = episode_store.get_episode(frozen_candidate.source_episode_id)
        if episode is None:
            raise LookupError(
                "Frozen Candidate source Episode is missing from its store"
            )
        stored_episode = episode
    else:
        episode = campaign_episode
        evidence_artifact = campaign["evidence_artifact"]
        evidence_path = (
            artifact_root / episode.run_id / evidence_artifact["relative_path"]
        )
        _write_json(evidence_path, evidence_artifact["content"], immutable=True)
        episode_store = EvolutionEpisodeStore(store_root)
        candidate_store = CandidateStore(store_root)
        stored_episode = episode_store.get_episode(episode.episode_id)
        if stored_episode is not None:
            current = episode.model_dump(mode="json", exclude={"created_at"})
            recorded = stored_episode.model_dump(mode="json", exclude={"created_at"})
            if current != recorded:
                raise RuntimeError(
                    "Stored Episode conflicts with the Campaign definition"
                )
            episode = stored_episode

    budget = CampaignBudget(args.token_cap)
    replay_llm = router.resolve_tier(args.replay_tier)
    stable_definition = registry.get(episode.target_skill_id)
    if stable_definition is None:
        raise RuntimeError(f"Stable Skill is missing: {episode.target_skill_id}")
    preflight_rows: list[dict[str, Any]] = []
    for raw in campaign["fixtures"]:
        fixture = _fixture_from_raw(
            raw,
            episode=episode,
            scenario_type=scenario_type,
            fixture_id=f"{raw['fixture_id']}-stable-headroom",
            target_metric=target_metric,
            invariants=invariants,
            arm_token_cap=args.arm_token_cap,
        )
        result = await _execute_fixture(
            scenario_type=scenario_type,
            fixture=fixture,
            instructions=stable_definition.load_instructions(),
            arm="stable",
            llm=replay_llm,
            budget=budget,
            max_tokens=args.replay_max_tokens,
        )
        hard_checks_passed = all(
            result.hard_checks[name] for name in fixture.invariants
        )
        preflight_rows.append(
            {
                "fixture_id": raw["fixture_id"],
                "fixture_role": str(raw.get("fixture_role") or "target"),
                "target_metric": target_metric,
                "metric_value": result.metrics[target_metric],
                "metrics": result.metrics,
                "hard_checks": result.hard_checks,
                "output": result.output,
                "tokens": result.total_tokens,
                "headroom_available": (
                    str(raw.get("fixture_role") or "target") == "target"
                    and (
                        result.metrics[target_metric] > 0
                        or not hard_checks_passed
                    )
                ),
            }
        )
    if args.preflight_only:
        if stored_episode is None:
            episode_store.write_run(episode.run_id, [episode])
        payload = {
            "schema_version": "1.0",
            "campaign_id": campaign_id,
            "campaign_file": str(campaign_path),
            "scenario_type": scenario_type,
            "target_metric": target_metric,
            "stable_preflight": preflight_rows,
            "candidate": {
                "candidate_id": "",
                "status": "preflight_only",
                "reused_existing": False,
                "generator_model_id": "",
                "verifier_model_id": "",
                "recorded_input_tokens": 0,
                "recorded_output_tokens": 0,
                "charged_generation_tokens_this_run": 0,
                "final_reasons": [
                    "Stable-only diagnostic; Candidate generation and replay were disabled."
                ],
                "store_path": "",
            },
            "replays": [],
            "all_replays_ready_for_shadow": False,
            "token_usage": budget.spent,
            "token_cap": budget.cap,
            "stable_skill_mutation_enabled": False,
            "production_loading_enabled": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "campaign-result.json", payload)
        (output_dir / "report.md").write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    if (
        frozen_candidate is None
        and not any(item["headroom_available"] for item in preflight_rows)
    ):
        if stored_episode is None:
            episode_store.write_run(episode.run_id, [episode])
        payload = {
            "schema_version": "1.0",
            "campaign_id": campaign_id,
            "campaign_file": str(campaign_path),
            "scenario_type": scenario_type,
            "target_metric": target_metric,
            "stable_preflight": preflight_rows,
            "candidate": {
                "candidate_id": "",
                "status": "skipped_no_headroom",
                "reused_existing": False,
                "generator_model_id": "",
                "verifier_model_id": "",
                "recorded_input_tokens": 0,
                "recorded_output_tokens": 0,
                "charged_generation_tokens_this_run": 0,
                "final_reasons": [
                    "Stable passed every target Fixture at the metric boundary; "
                    "Candidate generation was skipped."
                ],
                "store_path": "",
            },
            "replays": [],
            "all_replays_ready_for_shadow": False,
            "token_usage": budget.spent,
            "token_cap": budget.cap,
            "stable_skill_mutation_enabled": False,
            "production_loading_enabled": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "campaign-result.json", payload)
        (output_dir / "report.md").write_text(
            _markdown(payload), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    candidate_config = runtime.skill_evolution.candidate_generation.model_copy(
        update={
            "enabled": True,
            "max_total_tokens_per_candidate": args.candidate_token_cap,
        }
    )
    generator_llm = router.resolve_tier(candidate_config.generator_tier)
    semantic_verifier_llm = router.resolve_tier(candidate_config.verifier_tier)
    candidate_service = CandidateEvolutionService(
        registry=registry,
        episode_store=episode_store,
        candidate_store=candidate_store,
        artifact_root=artifact_root,
        generator_llm=generator_llm,
        verifier_llm=semantic_verifier_llm,
        config=candidate_config,
    )
    if frozen_candidate is not None:
        generated = CandidateGenerationResult(
            record=frozen_candidate,
            reused_existing=True,
            store_path=str(frozen_candidate_path),
        )
    else:
        failure_summaries: list[dict[str, Any]] = []
        for item in preflight_rows:
            if not item["headroom_available"]:
                continue
            diagnostics = {
                key: value
                for key, value in item["output"].items()
                if key not in {"cells", "entities", "rationale"} and value
            }
            failure_summaries.append(
                {
                    "fixture_id": item["fixture_id"],
                    "target_metric": item["target_metric"],
                    "metric_value": item["metric_value"],
                    "metrics": item["metrics"],
                    "failed_hard_checks": [
                        name
                        for name, passed in item["hard_checks"].items()
                        if not passed
                    ],
                    "diagnostics": diagnostics,
                }
            )
        observation_relative_path = (
            "details/skill-evolution/stable-headroom-observation.json"
        )
        observation_record_id = f"{episode.episode_id}-stable-observation"
        observation_path = artifact_root / episode.run_id / observation_relative_path
        _write_json(
            observation_path,
            {
                "observation_id": observation_record_id,
                "failures": failure_summaries,
            },
            immutable=True,
        )
        observed_episode = episode.model_copy(
            update={
                "episode_id": observation_record_id,
                "local_context": {
                    **episode.local_context,
                    "summary": (
                        "Stable local replay failed: "
                        + ", ".join(
                            sorted(
                                {
                                    name
                                    for item in failure_summaries
                                    for name in item["failed_hard_checks"]
                                }
                            )
                        )
                        + ". Use the linked Stable observation diagnostics as the "
                        "authoritative failure evidence."
                    ),
                    "stable_replay_failures": failure_summaries,
                },
                "local_result": {
                    **episode.local_result,
                    target_metric: sum(
                        float(item["metric_value"]) for item in failure_summaries
                    ),
                    "stable_headroom_fixture_count": len(failure_summaries),
                    "failed_hard_checks": sorted(
                        {
                            name
                            for item in failure_summaries
                            for name in item["failed_hard_checks"]
                        }
                    ),
                },
                "evidence_refs": [
                    EvolutionEvidenceReference(
                        artifact_path=observation_relative_path,
                        record_type="stable_headroom_observation",
                        record_id=observation_record_id,
                    ),
                ],
                "replay_spec": {
                    **episode.replay_spec,
                    "artifact_paths": [observation_relative_path],
                    "record_ids": [observation_record_id],
                },
            }
        )
        stored_observed = episode_store.get_episode(observed_episode.episode_id)
        if stored_observed is None:
            episode_store.write_run(observed_episode.run_id, [observed_episode])
            episode = observed_episode
        else:
            current = observed_episode.model_dump(
                mode="json", exclude={"created_at"}
            )
            recorded = stored_observed.model_dump(
                mode="json", exclude={"created_at"}
            )
            if current != recorded:
                raise RuntimeError(
                    "Stored observed Episode conflicts with Stable preflight"
                )
            episode = stored_observed
        generated = await candidate_service.generate_for_episode(episode.episode_id)
    generation_tokens = (
        0
        if generated.reused_existing
        else generated.record.total_input_tokens + generated.record.total_output_tokens
    )
    budget.charge_existing(generation_tokens)
    replay_rows: list[dict[str, Any]] = []

    if generated.record.status == "accept_for_replay":
        coevolution = VerifierCoEvolutionService(
            candidate_service=candidate_service,
            candidate_store=candidate_store,
            episode_store=episode_store,
            verifier_store=VerifierEvolutionStore(store_root),
            generator_llm=semantic_verifier_llm,
            config=VerifierCoEvolutionRuntimeConfig(enabled=True),
        )
        replay_llm = router.resolve_tier(args.replay_tier)

        async def executor(
            fixture: LocalMicroFixture, instructions: str, arm: str
        ) -> LocalReplayArmResult:
            return await _execute_fixture(
                scenario_type=scenario_type,
                fixture=fixture,
                instructions=instructions,
                arm=arm,
                llm=replay_llm,
                budget=budget,
                max_tokens=args.replay_max_tokens,
            )

        runner = LocalSkillReplayRunner(
            materializer=CandidateSkillMaterializer(
                registry=registry,
                replay_root=store_root / "local-replays",
            ),
            verifier_service=coevolution,
            executor=executor,
        )
        for raw in campaign["fixtures"]:
            fixture_id = str(raw["fixture_id"])
            if args.replay_id_suffix:
                if not args.replay_id_suffix.replace("-", "").replace("_", "").isalnum():
                    raise ValueError("replay_id_suffix contains unsafe characters")
                fixture_id = f"{fixture_id}-{args.replay_id_suffix}"
            fixture = _fixture_from_raw(
                raw,
                episode=episode,
                scenario_type=scenario_type,
                fixture_id=fixture_id,
                target_metric=target_metric,
                invariants=invariants,
                arm_token_cap=args.arm_token_cap,
            )
            execution = await runner.run(
                fixture=fixture,
                candidate=generated.record,
                episode=episode,
            )
            observation = execution.result.observation
            replay_rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "fixture_role": fixture.fixture_role,
                    "replay_id": execution.replay_id,
                    "status": execution.result.status,
                    "oracle_decision": execution.result.oracle_result.decision,
                    "target_metric": target_metric,
                    "stable_metric": observation.stable_metrics[target_metric],
                    "candidate_metric": observation.candidate_metrics[target_metric],
                    "stable_duplicate_queries": observation.stable_metrics.get(
                        "duplicate_queries", 0.0
                    ),
                    "candidate_duplicate_queries": observation.candidate_metrics.get(
                        "duplicate_queries", 0.0
                    ),
                    "stable_tokens": observation.stable_metrics["token_cost"],
                    "candidate_tokens": observation.candidate_metrics["token_cost"],
                    "candidate_hard_checks": observation.hard_checks,
                    "verifier_gap_id": execution.result.verifier_gap_id,
                    "paths": {
                        "observation": execution.observation_path,
                        "stable_result": execution.stable_result_path,
                        "candidate_result": execution.candidate_result_path,
                    },
                }
            )

    payload = {
        "schema_version": "1.0",
        "campaign_id": campaign_id,
        "campaign_file": str(campaign_path),
        "scenario_type": scenario_type,
        "target_metric": target_metric,
        "stable_preflight": preflight_rows,
        "candidate": {
            "candidate_id": generated.record.candidate_id,
            "status": generated.record.status,
            "reused_existing": generated.reused_existing,
            "generator_model_id": generated.record.generator_model_id,
            "verifier_model_id": generated.record.verifier_model_id,
            "recorded_input_tokens": generated.record.total_input_tokens,
            "recorded_output_tokens": generated.record.total_output_tokens,
            "charged_generation_tokens_this_run": generation_tokens,
            "final_reasons": generated.record.final_reasons,
            "store_path": generated.store_path,
        },
        "replays": replay_rows,
        "all_replays_ready_for_shadow": bool(replay_rows)
        and all(item["status"] == "ready_for_shadow" for item in replay_rows),
        "token_usage": budget.spent,
        "token_cap": budget.cap,
        "stable_skill_mutation_enabled": False,
        "production_loading_enabled": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "campaign-result.json", payload)
    (output_dir / "report.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        default="evaluation/fixtures/skill_evolution/live-duplicate-query-20260904-v1.json",
    )
    parser.add_argument("--config", default="config/runtime.yaml")
    parser.add_argument(
        "--store-root",
        default=".runtime/skill-evolution-live-duplicate-query-20260904-v1",
    )
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--output-dir",
        default="output/skill-micro-eval/live-duplicate-query-20260904-v1",
    )
    parser.add_argument("--token-cap", type=int, default=100000)
    parser.add_argument("--candidate-token-cap", type=int, default=40000)
    parser.add_argument("--arm-token-cap", type=int, default=12000)
    parser.add_argument("--replay-max-tokens", type=int, default=1200)
    parser.add_argument("--replay-tier", default="craft")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run Stable fixtures only; never generate or replay a Candidate.",
    )
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--replay-id-suffix", default="")
    parser.add_argument(
        "--candidate-file",
        default="",
        help="Replay one frozen accept_for_replay Candidate without regenerating it.",
    )
    args = parser.parse_args()
    if not 1 <= args.token_cap <= 1_000_000:
        parser.error("--token-cap must be between 1 and 1,000,000")
    if args.candidate_token_cap >= args.token_cap:
        parser.error("--candidate-token-cap must be below --token-cap")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
