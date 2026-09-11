"""Generic, domain-neutral acceptance tests for the SearchOS M1-M12 layer."""

import asyncio
import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from deepresearch.context_projection import project_context
from deepresearch.failure_memory import (
    active_failures,
    failures_from_fetch_traces,
    remember_failure,
)
from deepresearch.frontier import FrontierCommit, run_continuous_frontier
from deepresearch.plan_policy import ready_tasks
from deepresearch.schemas import (
    ClaimEvidenceLink,
    ClaimRecord,
    EvidenceBinding,
    EvidenceEdge,
    EvidenceExtractionResult,
    EvidenceRecord,
    EntityCandidate,
    ResearchTask,
    SearchSchemaDefinition,
    SearchFieldDefinition,
    SourceRecord,
    SourceSnapshot,
    TaskBudget,
)
from deepresearch.search_sensors import evaluate_progress
from deepresearch.search_state import (
    ObservationValidationError,
    calculate_dual_coverage,
    commit_observation_delta,
    prune_invalid_observation_items,
    promote_entity_candidate,
    reconcile_schema_revision,
    schema_from_plan,
    source_lead_from_legacy,
)
from deepresearch.observation_middleware import observations_from_extraction
from deepresearch.skill_loader import discover_skills
from deepresearch.skill_registry import SkillRegistry
from deepresearch.time_utils import now


def _task(task_id: str, *, depends_on: list[str] | None = None, consumes: list[str] | None = None):
    return ResearchTask(
        task_id=task_id,
        objective=task_id,
        research_dimension=task_id,
        depends_on=depends_on or [],
        consumes_artifact_roles=consumes or [],
        queries=[task_id],
        budget=TaskBudget(max_searches=3, max_fetches=3, max_wall_time_seconds=30),
    )


def _source(source_id: str = "src-1", content: str = "The value is 42.") -> dict:
    digest = hashlib.sha256(content.encode()).hexdigest()
    return {
        "source_id": source_id,
        "source_kind": "web_page",
        "title": "A source",
        "canonical_uri": f"https://example.test/{source_id}",
        "provider": "test",
        "fetched_at": "2026-08-10T12:00:00+08:00",
        "fetch_status": "full",
        "http_status": 200,
        "mime_type": "text/plain",
        "content_hash": digest,
        "quality_tier": 2,
        "quality_reason": "generic test source",
        "provenance": {"independent_domain": "example.test"},
    }


def _schema() -> SearchSchemaDefinition:
    return SearchSchemaDefinition.model_validate(
        {
            "schema_id": "schema-test",
            "tables": [
                {
                    "table_id": "entities",
                    "row_entity_type": "item",
                    "fields": [
                        {"field_key": "value", "label": "Value", "required": True},
                        {"field_key": "status", "label": "Status", "required": True},
                    ],
                }
            ],
            "scope_contract": {"target_description": "in-scope items"},
        }
    )


def _entity(entity_id: str = "entity-1", *, status: str = "materialized") -> dict:
    return EntityCandidate(
        entity_id=entity_id,
        entity_type="item",
        canonical_label=entity_id,
        anchor_evidence_ids=["ev-1"],
        scope_status="in_scope",
        materialization_status=status,
    ).model_dump(mode="json")


def _evidence(*, quote: str = "The value is 42.") -> dict:
    return EvidenceRecord(
        evidence_id="ev-1",
        source_id="src-1",
        quote=quote,
        quote_hash=hashlib.sha256(quote.encode()).hexdigest(),
        location={"kind": "text", "value": "0"},
    ).model_dump(mode="json")


def test_m1_dynamic_schema_relation_and_revision_preserve_artifacts():
    schema = _schema()
    revised = schema.model_copy(
        update={
            "tables": [
                schema.tables[0].model_copy(
                    update={
                        "fields": [
                            *schema.tables[0].fields,
                            SearchFieldDefinition(field_key="category", label="Category"),
                        ]
                    }
                )
            ]
        }
    )
    current, revision = reconcile_schema_revision(
        schema, revised, preserved_artifact_ids=["src-1", "ev-1"]
    )
    assert current.version == 2
    assert revision is not None
    assert revision.preserved_artifact_ids == ["src-1", "ev-1"]
    with pytest.raises(ValueError, match="unknown_search_relation_table"):
        SearchSchemaDefinition.model_validate(
            {
                "schema_id": "bad",
                "relations": [
                    {
                        "relation_id": "r",
                        "from_table": "missing",
                        "from_field": "entity_id",
                        "to_table": "missing",
                        "to_field": "entity_id",
                    }
                ],
            }
        )


def test_m1_legacy_lead_cannot_be_promoted_without_entity_evidence():
    lead = source_lead_from_legacy(
        {
            "lead_id": "lead-1",
            "task_id": "task-1",
            "title": "A report",
            "lead_type": "report",
            "artifact_role": "target_entity",
        }
    )
    assert lead.lead_id == "lead-1"
    assert not hasattr(lead, "entity_id")
    assert promote_entity_candidate(
        EntityCandidate(
            entity_id="e",
            entity_type="item",
            canonical_label="E",
            scope_status="in_scope",
        ),
        _schema(),
        known_evidence_ids=set(),
    ) is None


def test_m2_m3_m12_quote_binding_graph_and_idempotent_commit():
    claim = ClaimRecord(claim_id="claim-1", text="The value is 42.").model_dump(mode="json")
    link = ClaimEvidenceLink(
        link_id="link-1", claim_id="claim-1", evidence_id="ev-1"
    ).model_dump(mode="json")
    binding = EvidenceBinding(
        binding_id="binding-1",
        evidence_id="ev-1",
        entity_id="entity-1",
        field_key="value",
        proposed_value=42,
    ).model_dump(mode="json")
    edge = EvidenceEdge(
        edge_id="edge-1",
        source_node_id="ev-1",
        target_node_id="entity-1",
        relation="supports",
        evidence_id="ev-1",
    ).model_dump(mode="json")
    delta = {
        "delta_id": "delta-1",
        "task_id": "task-1",
        "source_records": [_source()],
        "source_snapshots": [
            SourceSnapshot(
                source_id="src-1",
                content="The value is 42.",
                content_hash=hashlib.sha256(b"The value is 42.").hexdigest(),
            ).model_dump(mode="json")
        ],
        "entity_candidates": [_entity()],
        "evidence_records": [_evidence()],
        "claims": [claim],
        "links": [link],
        "bindings": [binding],
        "graph_edges": [edge],
    }
    first = commit_observation_delta({"search_schemas": [_schema().model_dump(mode="json")]},
                                     __import__("deepresearch.schemas", fromlist=["ObservationDelta"]).ObservationDelta.model_validate(delta))
    patched = first["state_patch"]
    second = commit_observation_delta(patched, __import__("deepresearch.schemas", fromlist=["ObservationDelta"]).ObservationDelta.model_validate(delta))
    assert second["already_committed"] is True
    assert len(patched["evidence_bindings"]) == 1
    invalid = __import__("deepresearch.schemas", fromlist=["ObservationDelta"]).ObservationDelta.model_validate(
        {**delta, "delta_id": "delta-invalid", "evidence_records": [_evidence(quote="not present")]} 
    )
    with pytest.raises(ObservationValidationError, match="quote_not_in_snapshot"):
        commit_observation_delta({"search_schemas": [_schema().model_dump(mode="json")]}, invalid)


def test_invalid_evidence_pruning_preserves_unrelated_claims():
    delta = __import__(
        "deepresearch.schemas", fromlist=["ObservationDelta"]
    ).ObservationDelta.model_validate(
        {
            "delta_id": "delta-prune",
            "task_id": "task-prune",
            "source_records": [_source()],
            "source_snapshots": [
                SourceSnapshot(
                    source_id="src-1",
                    content="The value is 42.",
                    content_hash=hashlib.sha256(b"The value is 42.").hexdigest(),
                ).model_dump(mode="json")
            ],
            "evidence_records": [
                _evidence(),
                {**_evidence(quote="not present"), "evidence_id": "ev-bad"},
            ],
            "claims": [
                ClaimRecord(claim_id="claim-valid", text="Valid claim."),
                ClaimRecord(claim_id="claim-bad", text="Bad claim."),
            ],
            "links": [
                ClaimEvidenceLink(
                    link_id="link-valid",
                    claim_id="claim-valid",
                    evidence_id="ev-1",
                ),
                ClaimEvidenceLink(
                    link_id="link-bad",
                    claim_id="claim-bad",
                    evidence_id="ev-bad",
                ),
            ],
        }
    )
    pruned, report = prune_invalid_observation_items({}, delta)
    assert [item.evidence_id for item in pruned.evidence_records] == ["ev-1"]
    assert [item.claim_id for item in pruned.claims] == ["claim-valid"]
    assert [item.link_id for item in pruned.links] == ["link-valid"]
    assert report["rejected_claim_ids"] == ["claim-bad"]
    committed = commit_observation_delta({}, pruned)
    assert [item["claim_id"] for item in committed["state_patch"]["claims"]] == [
        "claim-valid"
    ]


def test_invalid_evidence_pruning_keeps_claim_with_another_valid_link():
    delta = __import__(
        "deepresearch.schemas", fromlist=["ObservationDelta"]
    ).ObservationDelta.model_validate(
        {
            "delta_id": "delta-multi-support",
            "task_id": "task-prune",
            "source_records": [_source()],
            "source_snapshots": [
                SourceSnapshot(
                    source_id="src-1",
                    content="The value is 42.",
                    content_hash=hashlib.sha256(b"The value is 42.").hexdigest(),
                ).model_dump(mode="json")
            ],
            "evidence_records": [
                _evidence(),
                {**_evidence(quote="not present"), "evidence_id": "ev-bad"},
            ],
            "claims": [ClaimRecord(claim_id="claim-1", text="The value is 42.")],
            "links": [
                ClaimEvidenceLink(
                    link_id="link-valid", claim_id="claim-1", evidence_id="ev-1"
                ),
                ClaimEvidenceLink(
                    link_id="link-bad", claim_id="claim-1", evidence_id="ev-bad"
                ),
            ],
        }
    )
    pruned, report = prune_invalid_observation_items({}, delta)
    assert [item.claim_id for item in pruned.claims] == ["claim-1"]
    assert [item.link_id for item in pruned.links] == ["link-valid"]
    assert report["rejected_claim_ids"] == []


def test_m4_cell_coverage_ignores_unmaterialized_and_unknown_fields():
    state = {
        "search_schemas": [_schema().model_dump(mode="json")],
        "entity_candidates": [_entity(), _entity("entity-2", status="discovered")],
        "evidence_bindings": [
            {"binding_id": "b1", "evidence_id": "ev-1", "entity_id": "entity-1", "field_key": "value", "status": "supported"},
            {"binding_id": "b2", "evidence_id": "ev-1", "entity_id": "entity-2", "field_key": "status", "status": "supported"},
            {"binding_id": "b3", "evidence_id": "ev-1", "entity_id": "entity-1", "field_key": "not_a_field", "status": "supported"},
        ],
        "coverage_changes": [
            {"change_id": "c1", "entity_id": "entity-1", "field_key": "status", "new_status": "unreachable"}
        ],
        "scope_completeness": {"status": "open"},
    }
    cell, scope = calculate_dual_coverage(state)
    assert cell.materialized_cell_count == 2
    assert cell.filled_cell_count == 1
    assert cell.unreachable_cell_count == 1
    assert cell.ratio == 0.5
    assert scope.status == "open"
    audited = {**state, "scope_completeness": {"status": "audited_complete"}}
    _, audited_scope = calculate_dual_coverage(audited)
    assert audited_scope.status == "audited_complete"


def test_m5_partial_gather_dependency_can_recover_missing_artifact_role():
    a = _task("a")
    b = _task("b")
    downstream = _task("down", depends_on=["a", "b"], consumes=["source_document", "target_entity"])
    runtime = [
        {"task_id": "a", "status": "partial", "dependency_usable": True, "produced_artifact_roles": ["source_document"]},
        {"task_id": "b", "status": "partial", "dependency_usable": True, "produced_artifact_roles": ["target_entity"]},
        {"task_id": "down", "status": "pending"},
    ]
    ready, blocked = ready_tasks([a, b, downstream], runtime)
    assert [item.task_id for item in ready] == ["down"]
    runtime[1]["produced_artifact_roles"] = []
    ready, blocked = ready_tasks([a, b, downstream], runtime)
    assert [item.task_id for item in ready] == ["down"]
    assert blocked == []


@pytest.mark.parametrize("task_kind", ["analyze", "verify"])
def test_m5_usable_dependency_role_mismatch_is_advisory(task_kind):
    origin = _task("origin")
    downstream = _task("down", depends_on=["origin"], consumes=["field_evidence"])
    downstream = downstream.model_copy(update={"task_kind": task_kind})
    runtime = [
        {
            "task_id": "origin",
            "status": "partial",
            "dependency_usable": True,
            "produced_artifact_roles": ["source_document"],
        },
        {"task_id": "down", "status": "pending"},
    ]
    ready, blocked = ready_tasks([origin, downstream], runtime)
    assert [item.task_id for item in ready] == ["down"]
    assert blocked == []


def test_m5_root_gather_can_start_without_upstream_artifact_roles():
    root = _task("root", consumes=["target_entity"])
    ready, blocked = ready_tasks(
        [root], [{"task_id": "root", "status": "pending"}]
    )
    assert [item.task_id for item in ready] == ["root"]
    assert blocked == []


@pytest.mark.asyncio
async def test_m6_continuous_frontier_refills_slot_by_dependency_layer():
    root_a, root_b = _task("a"), _task("b")
    child = _task("c", depends_on=["b"])
    runtime = [{"task_id": key, "status": "pending"} for key in ("a", "b", "c")]
    committed: list[str] = []
    a_finished = asyncio.Event()
    c_started = asyncio.Event()
    c_started_before_a_finished = False

    async def execute(task):
        nonlocal c_started_before_a_finished
        if task.task_id == "c":
            c_started_before_a_finished = not a_finished.is_set()
            c_started.set()
            await asyncio.sleep(0.005)
        elif task.task_id == "a":
            await asyncio.wait_for(c_started.wait(), timeout=0.5)
            await asyncio.sleep(0.005)
        else:
            await asyncio.sleep(0.005)
        if task.task_id == "a":
            a_finished.set()
        return {"contributions": [{"task_id": task.task_id, "status": "completed", "usable": True}]}

    async def commit(task, result):
        committed.append(task.task_id)
        return FrontierCommit(
            runtime_update={"task_id": task.task_id, "status": "completed", "dependency_usable": True},
        )

    result = await run_continuous_frontier(
        tasks=[root_a, root_b, child],
        runtime=runtime,
        max_concurrency=2,
        execute=execute,
        commit=commit,
    )
    assert result.launch_order[:2] == ["a", "b"]
    assert result.launch_order[-1] == "c"
    assert result.wave_task_ids[1] == ["a", "b"]
    assert result.wave_task_ids[2] == ["c"]
    assert committed.index("b") < committed.index("c")
    assert c_started_before_a_finished


def test_m7_worker_projection_keeps_global_index_without_unrelated_full_payloads():
    state = {
        "run_id": "run-projection",
        "request": {},
        "scope": {},
        "search_schemas": [],
        "evidence": [
            {"evidence_id": "ev-relevant", "source_id": "src-relevant", "quote": "relevant", "requirement_ids": ["req-1"]},
            {"evidence_id": "ev-noise", "source_id": "src-noise", "quote": "noise", "requirement_ids": ["req-2"]},
        ],
        "sources": [
            {"source_id": "src-relevant", "title": "Relevant", "content": "large relevant body"},
            {"source_id": "src-noise", "title": "Noise", "content": "large unrelated body"},
        ],
        "claims": [],
        "entity_candidates": [],
        "source_leads": [],
        "evidence_bindings": [],
        "failure_memory": [],
    }
    projection = project_context(
        role="search_worker",
        state=state,
        task={"task_id": "task", "requirement_ids": ["req-1"]},
    )
    assert [item["source_id"] for item in projection["sources"]] == ["src-relevant"]
    assert "large unrelated body" not in str(projection)
    assert {item["artifact_id"] for item in projection["artifact_index"]} == {
        "ev-relevant", "ev-noise", "src-relevant", "src-noise"
    }


def test_m8_extraction_observation_promotes_only_quote_anchored_entities():
    task = _task("extract")
    source = SourceRecord.model_validate(_source())
    evidence = [EvidenceRecord.model_validate(_evidence())]
    parsed = EvidenceExtractionResult.model_validate(
        {
            "artifact_role": "field_evidence",
            "target_entity_type": "item",
            "claims": [
                {
                    "text": "The value is 42.",
                    "quote": "The value is 42.",
                    "entity_id": "entity-1",
                    "entity_label": "Item 1",
                    "entity_type": "item",
                    "entity_scope_status": "in_scope",
                    "field_key": "value",
                    "field_value": 42,
                }
            ],
        }
    )
    result = observations_from_extraction(
        run_id="run-m8",
        task=task,
        source=source,
        parsed=parsed,
        accepted_candidates=[parsed.claims[0].model_dump()],
        evidence=evidence,
        schema=_schema(),
    )
    assert result["entity_candidates"][0]["materialization_status"] == "materialized"
    assert result["evidence_bindings"][0]["field_key"] == "value"
    assert result["evidence_edges"]


def test_extraction_recovers_unambiguous_task_level_binding():
    task = _task("extract-single-target").model_copy(
        update={
            "target_entity_ids": ["entity-1"],
            "target_field_keys": ["value"],
            "requirement_ids": ["req-value"],
            "produces_requirement_ids": ["req-value"],
        }
    )
    source = SourceRecord.model_validate(_source())
    evidence = EvidenceRecord.model_validate(
        {**_evidence(), "requirement_ids": ["req-value"]}
    )
    parsed = EvidenceExtractionResult.model_validate(
        {
            "artifact_role": "field_evidence",
            "target_entity_type": "item",
            "claims": [
                {
                    "text": "The value is 42.",
                    "quote": "The value is 42.",
                    "requirement_ids": ["req-value"],
                    "field_value": 42,
                }
            ],
        }
    )

    result = observations_from_extraction(
        run_id="run-single-target",
        task=task,
        source=source,
        parsed=parsed,
        accepted_candidates=[parsed.claims[0].model_dump()],
        evidence=[evidence],
        schema=_schema(),
    )

    assert result["entity_candidates"] == []
    assert len(result["evidence_bindings"]) == 1
    binding = result["evidence_bindings"][0]
    assert binding["entity_id"] == "entity-1"
    assert binding["field_key"] == "value"
    assert binding["table_id"] == "entities"
    assert binding["requirement_id"] == "req-value"
    assert "unambiguous task target" in binding["rationale"]


@pytest.mark.parametrize(
    ("target_entities", "target_fields", "claim_requirements", "evidence_requirements"),
    [
        (["entity-1", "entity-2"], ["value"], ["req-value"], ["req-value"]),
        (["entity-1"], ["value", "status"], ["req-value"], ["req-value"]),
        (["entity-1"], ["value"], ["req-other"], ["req-value"]),
        (["entity-1"], ["value"], ["req-value"], ["req-other"]),
    ],
)
def test_extraction_does_not_guess_ambiguous_task_level_binding(
    target_entities,
    target_fields,
    claim_requirements,
    evidence_requirements,
):
    task = _task("extract-ambiguous").model_copy(
        update={
            "target_entity_ids": target_entities,
            "target_field_keys": target_fields,
            "requirement_ids": ["req-value"],
            "produces_requirement_ids": ["req-value"],
        }
    )
    source = SourceRecord.model_validate(_source())
    evidence = EvidenceRecord.model_validate(
        {**_evidence(), "requirement_ids": evidence_requirements}
    )
    parsed = EvidenceExtractionResult.model_validate(
        {
            "artifact_role": "field_evidence",
            "target_entity_type": "item",
            "claims": [
                {
                    "text": "The value is 42.",
                    "quote": "The value is 42.",
                    "requirement_ids": claim_requirements,
                    "field_value": 42,
                }
            ],
        }
    )

    result = observations_from_extraction(
        run_id="run-ambiguous",
        task=task,
        source=source,
        parsed=parsed,
        accepted_candidates=[parsed.claims[0].model_dump()],
        evidence=[evidence],
        schema=_schema(),
    )

    assert result["evidence_bindings"] == []


def test_m9_sensor_requires_multiple_low_value_signals():
    base = {
        "entity_count": 1,
        "materialized_entity_count": 1,
        "bound_evidence_count": 1,
        "filled_cell_count": 1,
        "independent_source_count": 1,
        "duplicate_query_ratio": 0.9,
        "failed_fetch_ratio": 0.9,
    }
    two = evaluate_progress([base, base], minimum_windows=2, minimum_stop_signals=3)
    assert two["action"] != "consider_drain"
    three = evaluate_progress([base, base, base, base], minimum_windows=2, minimum_stop_signals=3)
    assert three["action"] == "consider_drain"


def test_m10_failure_memory_has_scope_ttl_and_recurrence():
    traces = [{"url": "https://example.test/a", "status": "failed", "error": "timeout", "dynamic_attempted": True}]
    records = failures_from_fetch_traces(run_id="run-f", task_id="task-a", traces=traces)
    assert records[0].expires_at is not None
    assert active_failures(records, task_id="task-a")
    assert not active_failures(records, task_id="task-b")
    updated, record = remember_failure(
        records,
        scope_signature=records[0].scope_signature,
        failure_type=records[0].failure_type,
        action_signature=records[0].action_signature,
        task_id="task-a",
        correction="try another access route",
    )
    assert record.recurrence_count == 2
    assert record.correction == "try another access route"
    expired_at = (now() - timedelta(seconds=1)).isoformat()
    _, expired = remember_failure(
        updated,
        scope_signature="run:old/task:a",
        failure_type="url_unreachable",
        action_signature="fetch:old",
        expires_at=expired_at,
    )
    assert not active_failures([expired])


def test_m11_skill_layers_are_role_and_profile_filterable(tmp_path):
    definitions = discover_skills(Path(__file__).resolve().parents[1])
    registry = SkillRegistry()
    for definition in definitions:
        registry.register(definition)
    assert registry.list(role="planner", profile_id="general_research", layers=["orchestrator"])
    assert registry.list(role="worker", profile_id="general_research", layers=["strategy"])
    assert registry.list(role="worker", profile_id="general_research", layers=["access"])
    assert not registry.list(role="planner", profile_id="general_research", executable_only=True)
