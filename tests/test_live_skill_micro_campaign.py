from __future__ import annotations

import json

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.skill_evolution import LocalMicroFixture
from tools.run_live_skill_micro_campaign import (
    CampaignBudget,
    _execute_analysis_chain,
    _execute_coverage_gap_planning,
    _execute_entity_discovery,
    _execute_entity_materialization,
    _execute_field_matrix,
    _execute_query_reformulation,
    _execute_query_selection,
    _execute_route_selection,
    _execute_route_selection_history,
    _execute_source_selection,
    _execute_task_contract_decomposition,
)


@pytest.mark.asyncio
async def test_coverage_gap_planning_hides_oracle_and_scores_missing_gap():
    fixture = _fixture(
        scenario="coverage_gap_planning",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Complete a dated event timeline.",
            "required_cells": [
                {"cell_key": "event-a", "requirement": "exact date and act"},
                {"cell_key": "event-b", "requirement": "exact date and act"},
            ],
            "current_coverage": [
                {"cell_key": "event-a", "status": "supported"},
                {"cell_key": "event-b", "status": "partial"},
            ],
            "oracle": {
                "expected_target_cell_keys": ["event-b"],
                "forbidden_target_cell_keys": ["event-a"],
                "allowed_scope_completeness_statuses": ["open"],
            },
        },
    )
    llm = RecordingJSONLLM(
        {
            "selected_cell_keys": [],
            "scope_completeness_status": "open",
            "rationale": "done",
        }
    )
    result = await _execute_coverage_gap_planning(
        fixture=fixture,
        instructions="Preserve literal scope.",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=200,
    )
    assert result.metrics["skill_execution_failures"] == 1
    assert result.hard_checks["all_material_gaps_targeted"] is False
    assert "oracle" not in llm.calls[0]["user"]


@pytest.mark.asyncio
async def test_analysis_chain_hides_oracle_and_scores_missing_premise():
    fixture = _fixture(
        scenario="analysis_chain",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Compare two liability routes.",
            "evidence_items": [
                {"evidence_id": "e1", "text": "Route A is strict."},
                {"evidence_id": "e2", "text": "Route B requires fault."},
            ],
            "analysis_requests": [
                {"request_id": "r1", "question": "Explain the difference."}
            ],
            "oracle": {
                "request_rules": [
                    {
                        "request_id": "r1",
                        "required_evidence_ids": ["e1", "e2"],
                        "must_contain_any_groups": [["strict"], ["fault"]],
                    }
                ]
            },
        },
    )
    llm = RecordingJSONLLM(
        {
            "analysis_units": [
                {
                    "request_id": "r1",
                    "premise_evidence_ids": ["e1"],
                    "reasoning": "Route A is strict.",
                    "conclusion": "They differ.",
                }
            ]
        }
    )
    result = await _execute_analysis_chain(
        fixture=fixture,
        instructions="Use verified claims.",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=200,
    )
    assert result.metrics["skill_execution_failures"] == 2
    assert result.hard_checks["required_premises_used"] is False
    assert result.hard_checks["reasoning_concepts_preserved"] is False
    assert "oracle" not in llm.calls[0]["user"]


class RecordingJSONLLM:
    model_id = "fixture-llm"

    def __init__(self, value: dict[str, object]):
        self.value = value
        self.calls: list[dict[str, object]] = []

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int
    ) -> tuple[dict[str, object], LLMResponse]:
        self.calls.append(
            {"system": system, "user": json.loads(user), "max_tokens": max_tokens}
        )
        return self.value, LLMResponse(
            content=json.dumps(self.value), input_tokens=20, output_tokens=10
        )


def _fixture(
    *, scenario: str, snapshot: dict[str, object], metric: str
) -> LocalMicroFixture:
    return LocalMicroFixture(
        fixture_id=f"fixture-{scenario}",
        target_skill_id="web-search",
        trigger_type=scenario,
        input_snapshot=snapshot,
        target_metric=metric,
        invariants=[],
        max_model_calls=1,
        max_tokens=1000,
    )


@pytest.mark.asyncio
async def test_query_selection_scores_duplicates_and_intent_preservation():
    fixture = _fixture(
        scenario="duplicate_query",
        metric="duplicate_queries",
        snapshot={
            "queries": [
                {"query_id": "q1", "query": "SkillX paper"},
                {"query_id": "q2", "query": "SkillX research paper"},
                {"query_id": "q3", "query": "CoEvoSkills paper"},
            ],
            "equivalence_groups": [["q1", "q2"], ["q3"]],
        },
    )
    llm = RecordingJSONLLM(
        {"selected_query_ids": ["q1", "q2", "q3"], "rationale": "test"}
    )

    result = await _execute_query_selection(
        fixture=fixture,
        instructions="deduplicate equivalent queries",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert result.metrics["duplicate_queries"] == 1
    assert result.hard_checks == {
        "valid_query_ids_only": True,
        "all_intents_preserved": True,
        "one_query_per_intent": False,
    }


@pytest.mark.asyncio
async def test_task_contract_decomposition_scores_local_partial_contract():
    fixture = _fixture(
        scenario="task_contract_decomposition",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Build a complete 21-trial matrix.",
            "current_coverage": {"known_entities": 0, "expected_entities": 21},
            "available_source": {
                "source_id": "source-ssass",
                "text": "SSaSS was a cluster-randomized trial in China.",
                "supported_field_keys": ["trial_identity", "country", "design"],
            },
            "oracle": {
                "required_field_keys": ["trial_identity", "design"],
                "allowed_completion_contracts": ["extract_entities", "fill_fields"],
            },
        },
    )
    llm = RecordingJSONLLM(
        {
            "task_contract": {
                "objective": "Extract the locally supported SSaSS row fields.",
                "completion_contract": "extract_entities",
                "source_ids": ["source-ssass"],
                "target_entity_ids": [],
                "target_field_keys": ["trial_identity", "country", "design"],
                "allow_partial_results": True,
                "accept_unresolved_fields": True,
                "queries": [],
            },
            "rationale": "Use the frozen source without expanding scope.",
        }
    )

    result = await _execute_task_contract_decomposition(
        fixture=fixture,
        instructions="Create independent evidence-oriented worker contracts.",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=500,
    )

    assert result.metrics["skill_execution_failures"] == 0
    assert all(result.hard_checks.values())
    assert "oracle" not in llm.calls[0]["user"]


@pytest.mark.asyncio
async def test_route_selection_hides_outcomes_and_scores_failed_family():
    fixture = _fixture(
        scenario="source_family_repeated_failure",
        metric="source_family_failures",
        snapshot={
            "prior_failed_source_families": ["publisher.example"],
            "available_routes": [
                {
                    "route_id": "same-family",
                    "source_family": "publisher.example",
                    "allowed": True,
                    "outcome": "failure",
                },
                {
                    "route_id": "open-copy",
                    "source_family": "repository.example",
                    "allowed": True,
                    "outcome": "success",
                },
            ],
        },
    )
    llm = RecordingJSONLLM(
        {"selected_route_ids": ["same-family", "open-copy"], "rationale": "test"}
    )

    result = await _execute_route_selection(
        fixture=fixture,
        instructions="switch source families after repeated failure",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    routes_seen_by_model = llm.calls[0]["user"]["available_routes"]
    assert all("outcome" not in route for route in routes_seen_by_model)
    assert result.metrics["source_family_failures"] == 1
    assert result.hard_checks["successful_route_reached"] is True


@pytest.mark.asyncio
async def test_route_selection_history_hides_oracle_and_scores_normalized_family():
    fixture = _fixture(
        scenario="source_family_repeated_failure",
        metric="source_family_failures",
        snapshot={
            "prior_attempts": [
                {
                    "url": "https://doi.org/10.1/example",
                    "redirect_host": "publisher.example",
                    "route_type": "direct",
                    "status": "failed",
                }
            ],
            "oracle_exhausted_source_families": ["publisher.example"],
            "available_routes": [
                {
                    "route_id": "same-family",
                    "source_family": "publisher.example",
                    "allowed": True,
                    "outcome": "failure",
                },
                {
                    "route_id": "open-copy",
                    "source_family": "repository.example",
                    "allowed": True,
                    "outcome": "success",
                },
            ],
        },
    )
    llm = RecordingJSONLLM(
        {"selected_route_ids": ["same-family", "open-copy"], "rationale": "test"}
    )

    result = await _execute_route_selection_history(
        fixture=fixture,
        instructions="Use bounded retries and allowed alternatives.",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert "oracle_exhausted_source_families" not in llm.calls[0]["user"]
    assert all(
        "outcome" not in route
        for route in llm.calls[0]["user"]["available_routes"]
    )
    assert result.metrics["source_family_failures"] == 1
    assert result.hard_checks["successful_route_reached"] is True


@pytest.mark.asyncio
async def test_entity_discovery_neutral_system_omits_behavioral_answer_hints():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Create one row per trial.",
            "source_documents": [
                {"source_id": "s1", "text": "Trial Alpha was randomized."}
            ],
            "gold_entities": [
                {
                    "entity_id": "trial-alpha",
                    "canonical_name": "Trial Alpha",
                    "aliases": [],
                    "source_ids": ["s1"],
                }
            ],
            "container_names": [],
            "expected_completeness_statuses": ["audited_complete"],
            "neutral_system": True,
        },
    )
    llm = RecordingJSONLLM(
        {
            "entities": [
                {
                    "canonical_name": "Trial Alpha",
                    "aliases": [],
                    "source_ids": ["s1"],
                }
            ],
            "scope_completeness_status": "audited_complete",
            "rationale": "test",
        }
    )

    result = await _execute_entity_discovery(
        fixture=fixture,
        instructions="Plan evidence-oriented tasks.",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert result.metrics["skill_execution_failures"] == 0
    assert "Exclude source" not in llm.calls[0]["system"]
    assert "Resolve aliases" not in llm.calls[0]["system"]


@pytest.mark.asyncio
async def test_source_selection_scores_repeated_domains_and_authority():
    fixture = _fixture(
        scenario="repeated_source_domain",
        metric="repeated_domain_rate",
        snapshot={
            "prior_selected_domains": ["example.org"],
            "selection_count": 2,
            "candidate_sources": [
                {
                    "source_id": "repeated",
                    "domain": "example.org",
                    "allowed": True,
                    "quality": "primary",
                },
                {
                    "source_id": "fresh",
                    "domain": "fresh.example",
                    "allowed": True,
                    "quality": "authoritative-copy",
                },
            ],
        },
    )
    llm = RecordingJSONLLM(
        {"selected_source_ids": ["repeated", "fresh"], "rationale": "test"}
    )

    result = await _execute_source_selection(
        fixture=fixture,
        instructions="avoid repeatedly selecting one domain",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert result.metrics["repeated_domain_rate"] == 0.5
    assert all(result.hard_checks.values())


@pytest.mark.asyncio
async def test_query_reformulation_hides_oracle_fields_and_scores_gain():
    fixture = _fixture(
        scenario="consecutive_no_evidence_gain",
        metric="zero_gain_searches",
        snapshot={
            "research_scope": "SkillX paper",
            "prior_queries": ["SkillX"],
            "candidate_queries": [
                {
                    "query_id": "exact-title",
                    "query": '"SkillX" agent skills paper',
                    "strategy": "exact title",
                    "outcome": "gain",
                    "in_scope": True,
                },
                {
                    "query_id": "topic-drift",
                    "query": "workplace skills 2026",
                    "strategy": "broad topic",
                    "outcome": "no_gain",
                    "in_scope": False,
                },
            ],
        },
    )
    llm = RecordingJSONLLM(
        {"selected_query_ids": ["exact-title"], "rationale": "test"}
    )

    result = await _execute_query_reformulation(
        fixture=fixture,
        instructions="reformulate after zero evidence gain",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    candidates_seen_by_model = llm.calls[0]["user"]["candidate_queries"]
    assert all("outcome" not in item for item in candidates_seen_by_model)
    assert all("in_scope" not in item for item in candidates_seen_by_model)
    assert all("strategy" not in item for item in candidates_seen_by_model)
    assert result.metrics == {"zero_gain_searches": 0.0, "evidence_gain": 1.0}
    assert all(result.hard_checks.values())


@pytest.mark.asyncio
async def test_entity_materialization_hides_gold_and_scores_container_error():
    fixture = _fixture(
        scenario="container_entity_confusion",
        metric="entity_errors",
        snapshot={
            "research_scope": "List each trial, not the review.",
            "source_documents": [
                {
                    "source_id": "review",
                    "title": "Review",
                    "text": "The review lists Trial A and Trial B.",
                }
            ],
            "candidate_entities": [
                {"entity_id": "review", "label": "Review", "kind": "review"},
                {"entity_id": "trial-a", "label": "Trial A", "kind": "trial"},
                {"entity_id": "trial-b", "label": "Trial B", "kind": "trial"},
            ],
            "gold_entity_ids": ["trial-a", "trial-b"],
            "container_entity_ids": ["review"],
            "required_document_entity_ids": [],
        },
    )
    llm = RecordingJSONLLM(
        {"selected_entity_ids": ["review", "trial-a"], "rationale": "test"}
    )

    result = await _execute_entity_materialization(
        fixture=fixture,
        instructions="preserve the literal entity scope",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    payload_seen_by_model = llm.calls[0]["user"]
    assert "gold_entity_ids" not in payload_seen_by_model
    assert "container_entity_ids" not in payload_seen_by_model
    assert "required_document_entity_ids" not in payload_seen_by_model
    assert result.metrics["entity_errors"] == 2
    assert result.hard_checks["container_entities_excluded"] is False
    assert result.hard_checks["target_entity_set_complete"] is False


@pytest.mark.asyncio
async def test_entity_discovery_hides_gold_and_scores_alias_and_container_errors():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "List each trial, not the review.",
            "source_documents": [
                {
                    "source_id": "review",
                    "title": "Trials Review",
                    "text": "The review lists Trial A and Trial B (TB).",
                }
            ],
            "gold_entities": [
                {
                    "entity_id": "trial-a",
                    "canonical_name": "Portfolio Selection",
                    "aliases": [],
                },
                {
                    "entity_id": "trial-b",
                    "canonical_name": "Trial B",
                    "aliases": ["TB"],
                },
            ],
            "container_names": ["Trials Review"],
        },
    )
    llm = RecordingJSONLLM(
        {
            "entities": [
                {"canonical_name": "Portfolio Selection (1952)", "aliases": []},
                {"canonical_name": "TB", "aliases": []},
                {"canonical_name": "Trials Review", "aliases": []},
            ],
            "rationale": "test",
        }
    )

    result = await _execute_entity_discovery(
        fixture=fixture,
        instructions="expand containers without materializing them",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    payload_seen_by_model = llm.calls[0]["user"]
    assert "gold_entities" not in payload_seen_by_model
    assert "container_names" not in payload_seen_by_model
    assert result.metrics["entity_recall"] == 1.0
    assert result.metrics["entity_precision"] == pytest.approx(2 / 3)
    assert result.metrics["skill_execution_failures"] == 1
    assert result.hard_checks["container_entities_excluded"] is False


@pytest.mark.asyncio
async def test_entity_discovery_scores_scope_completeness_without_leaking_gold():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Enumerate every eligible trial.",
            "source_documents": [
                {
                    "source_id": "review",
                    "title": "Review",
                    "text": (
                        "Trial A is eligible. Two additional eligible trials are "
                        "listed only in an unavailable supplement."
                    ),
                }
            ],
            "gold_entities": [
                {
                    "entity_id": "trial-a",
                    "canonical_name": "Trial A",
                    "aliases": [],
                }
            ],
            "container_names": ["Review"],
            "expected_completeness_statuses": ["open"],
        },
    )
    llm = RecordingJSONLLM(
        {
            "entities": [{"canonical_name": "Trial A", "aliases": []}],
            "scope_completeness_status": "audited_complete",
            "rationale": "test",
        }
    )

    result = await _execute_entity_discovery(
        fixture=fixture,
        instructions="preserve literal scope",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    payload_seen_by_model = llm.calls[0]["user"]
    assert "gold_entities" not in payload_seen_by_model
    assert "expected_completeness_statuses" not in payload_seen_by_model
    assert result.metrics["entity_recall"] == 1.0
    assert result.metrics["skill_execution_failures"] == 1.0
    assert result.hard_checks["scope_completeness_calibrated"] is False


@pytest.mark.asyncio
async def test_entity_discovery_scores_direct_membership_source_binding():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Enumerate eligible trials.",
            "source_documents": [
                {
                    "source_id": "roster",
                    "title": "Candidate roster",
                    "text": "The roster mentions Trial A as a candidate.",
                },
                {
                    "source_id": "eligibility",
                    "title": "Eligibility ruling",
                    "text": "Trial A meets all eligibility criteria.",
                },
            ],
            "gold_entities": [
                {
                    "entity_id": "trial-a",
                    "canonical_name": "Trial A",
                    "aliases": [],
                    "source_ids": ["eligibility"],
                }
            ],
            "container_names": ["Candidate roster"],
            "expected_completeness_statuses": ["audited_complete"],
        },
    )
    llm = RecordingJSONLLM(
        {
            "entities": [
                {
                    "canonical_name": "Trial A",
                    "aliases": [],
                    "source_ids": ["roster"],
                }
            ],
            "scope_completeness_status": "audited_complete",
        }
    )

    result = await _execute_entity_discovery(
        fixture=fixture,
        instructions="preserve literal scope",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert result.metrics["entity_recall"] == 1.0
    assert result.metrics["entity_source_binding_accuracy"] == 0.0
    assert result.metrics["skill_execution_failures"] == 1.0
    assert result.hard_checks["entity_sources_valid"] is False
    assert result.output["source_binding_error_names"] == ["Trial A"]


@pytest.mark.asyncio
async def test_field_matrix_hides_gold_and_scores_supported_conflict_and_nr():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Fill every trial field.",
            "entities": [{"entity_id": "trial-a", "name": "Trial A"}],
            "field_schema": [
                {"field_key": "randomized_n", "description": "Final randomized"},
                {"field_key": "withdrawals_n", "description": "Withdrawals"},
                {"field_key": "adverse_events", "description": "Serious events"},
            ],
            "source_documents": [
                {
                    "source_id": "main",
                    "title": "Main",
                    "text": "Trial A randomized 100 participants. Table 3 reports 7 withdrawals.",
                },
                {
                    "source_id": "appendix",
                    "title": "Appendix",
                    "text": "The appendix reports 8 withdrawals. Serious adverse events were not reported.",
                },
            ],
            "gold_cells": [
                {
                    "entity_id": "trial-a",
                    "field_key": "randomized_n",
                    "status": "supported",
                    "expected_values": [{"value": "100", "source_ids": ["main"]}],
                },
                {
                    "entity_id": "trial-a",
                    "field_key": "withdrawals_n",
                    "status": "conflict",
                    "expected_values": [
                        {"value": "7", "source_ids": ["main"]},
                        {"value": "8", "source_ids": ["appendix"]},
                    ],
                },
                {
                    "entity_id": "trial-a",
                    "field_key": "adverse_events",
                    "status": "nr",
                    "expected_values": [],
                },
            ],
            "neutral_system": True,
        },
    )
    llm = RecordingJSONLLM(
        {
            "cells": [
                {
                    "entity_id": "trial-a",
                    "field_key": "randomized_n",
                    "status": "supported",
                    "values": [
                        {
                            "value": "100",
                            "source_id": "main",
                            "quote": "Trial A randomized 100 participants.",
                        }
                    ],
                },
                {
                    "entity_id": "trial-a",
                    "field_key": "withdrawals_n",
                    "status": "conflict",
                    "values": [
                        {
                            "value": "7",
                            "source_id": "main",
                            "quote": "Table 3 reports 7 withdrawals.",
                        },
                        {
                            "value": "8",
                            "source_id": "appendix",
                            "quote": "The appendix reports 8 withdrawals.",
                        },
                    ],
                },
                {
                    "entity_id": "trial-a",
                    "field_key": "adverse_events",
                    "status": "nr",
                    "values": [],
                },
            ],
            "rationale": "test",
        }
    )

    result = await _execute_field_matrix(
        fixture=fixture,
        instructions="copy exact evidence quotes",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert "gold_cells" not in llm.calls[0]["user"]
    assert "Use conflict only" not in llm.calls[0]["system"]
    assert result.metrics == {
        "skill_execution_failures": 0.0,
        "cell_accuracy": 1.0,
        "cell_coverage": 1.0,
        "nr_correctness": 1.0,
    }
    assert all(result.hard_checks.values())


@pytest.mark.asyncio
async def test_field_matrix_records_noncontiguous_quote_diagnostics():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Extract the sample size.",
            "entities": [{"entity_id": "trial-a", "name": "Trial A"}],
            "field_schema": [
                {"field_key": "randomized_n", "description": "Final randomized"}
            ],
            "source_documents": [
                {
                    "source_id": "main",
                    "title": "Main",
                    "text": "Trial A reports a final randomized sample of 100.",
                }
            ],
            "gold_cells": [
                {
                    "entity_id": "trial-a",
                    "field_key": "randomized_n",
                    "status": "supported",
                    "expected_values": [{"value": "100", "source_ids": ["main"]}],
                }
            ],
        },
    )
    llm = RecordingJSONLLM(
        {
            "cells": [
                {
                    "entity_id": "trial-a",
                    "field_key": "randomized_n",
                    "status": "supported",
                    "values": [
                        {
                            "value": "100",
                            "source_id": "main",
                            "quote": "Trial A reports ... 100.",
                        }
                    ],
                }
            ]
        }
    )

    result = await _execute_field_matrix(
        fixture=fixture,
        instructions="copy exact evidence quotes",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert result.metrics["skill_execution_failures"] == 1
    assert result.hard_checks["quotes_exact_and_sources_valid"] is False
    assert result.output["invalid_evidence_details"] == [
        {
            "cell_key": "trial-a::randomized_n",
            "value": "100",
            "source_id": "main",
            "quote": "Trial A reports ... 100.",
            "reason": "quote_is_not_an_exact_contiguous_source_substring",
            "source_excerpt": "Trial A reports a final randomized sample of 100.",
        }
    ]


@pytest.mark.asyncio
async def test_field_matrix_can_require_quote_to_identify_target_entity():
    fixture = _fixture(
        scenario="skill_execution_failure",
        metric="skill_execution_failures",
        snapshot={
            "research_scope": "Extract each subsidiary ownership percentage.",
            "entities": [
                {"entity_id": "sub-beacon", "name": "Beacon Cloud GmbH"}
            ],
            "field_schema": [
                {"field_key": "ownership", "description": "Ownership percentage"}
            ],
            "source_documents": [
                {
                    "source_id": "annual-report",
                    "title": "Annual report",
                    "text": "The Group owned 72% of Aurora and 41% of Beacon Cloud GmbH.",
                }
            ],
            "gold_cells": [
                {
                    "entity_id": "sub-beacon",
                    "field_key": "ownership",
                    "status": "supported",
                    "expected_values": [
                        {
                            "value": "41%",
                            "source_ids": ["annual-report"],
                            "quote_must_contain_any": ["Beacon Cloud GmbH", "Beacon"],
                        }
                    ],
                }
            ],
        },
    )
    llm = RecordingJSONLLM(
        {
            "cells": [
                {
                    "entity_id": "sub-beacon",
                    "field_key": "ownership",
                    "status": "supported",
                    "values": [
                        {
                            "value": "41%",
                            "source_id": "annual-report",
                            "quote": "41%",
                        }
                    ],
                }
            ]
        }
    )

    result = await _execute_field_matrix(
        fixture=fixture,
        instructions="copy entity-specific exact evidence quotes",
        arm="stable",
        llm=llm,
        budget=CampaignBudget(10_000),
        max_tokens=100,
    )

    assert result.metrics["skill_execution_failures"] == 1
    assert result.output["invalid_evidence_details"][0]["reason"] == (
        "quote_does_not_identify_the_target_entity"
    )
