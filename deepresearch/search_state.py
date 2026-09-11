"""Typed, run-local search state and replay-safe observation commits.

Semantic choices are supplied by Planner/Worker models.  This module only
enforces referential integrity, quote anchoring, idempotency, and conservative
coverage accounting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from .ids import make_id, normalize_text
from .schemas import (
    CandidateLead,
    CellCoverage,
    EntityCandidate,
    EvidenceBinding,
    EvidenceEdge,
    EvidenceRecord,
    ObservationDelta,
    ScopeCompleteness,
    ScopeContract,
    SearchFieldDefinition,
    SearchRelationDefinition,
    SearchSchemaDefinition,
    SearchTableDefinition,
    SchemaRevision,
    SourceLead,
    SourceSnapshot,
)


class ObservationValidationError(ValueError):
    """Raised when an ObservationDelta would create an inconsistent state."""


def _merge_by_id(
    existing: Iterable[Mapping[str, Any]],
    incoming: Iterable[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*existing, *incoming]:
        identity = str(item.get(key) or "")
        if identity:
            merged[identity] = dict(item)
    return list(merged.values())


def _append_only_by_id(
    existing: Iterable[Mapping[str, Any]],
    incoming: Iterable[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    output = [dict(item) for item in existing]
    by_id = {str(item.get(key) or ""): item for item in output if item.get(key)}
    for raw in incoming:
        item = dict(raw)
        identity = str(item.get(key) or "")
        if not identity:
            continue
        current = by_id.get(identity)
        if current is None:
            output.append(item)
            by_id[identity] = item
        elif current != item:
            versioned_id = make_id("revision", identity, item)
            if versioned_id not in by_id:
                versioned = {
                    **item,
                    key: versioned_id,
                }
                output.append(versioned)
                by_id[versioned_id] = versioned
    return output


def _legacy_fields(plan: Mapping[str, Any]) -> list[SearchFieldDefinition]:
    """Adapt an old coverage blueprint without inventing domain semantics."""

    blueprint = plan.get("coverage_blueprint") or {}
    raw_fields = blueprint.get("fields") or blueprint.get("required_fields") or []
    fields: list[SearchFieldDefinition] = []
    for item in raw_fields:
        if isinstance(item, Mapping):
            key = normalize_text(
                str(item.get("field_key") or item.get("key") or item.get("name") or "")
            )
            label = normalize_text(str(item.get("label") or item.get("name") or key))
            description = normalize_text(str(item.get("description") or ""))
        else:
            key = normalize_text(str(item))
            label = key
            description = ""
        if key and key not in {field.field_key for field in fields}:
            fields.append(
                SearchFieldDefinition(
                    field_key=key,
                    label=label or key,
                    description=description,
                )
            )
    return fields


def schema_from_plan(
    *,
    run_id: str,
    plan: Mapping[str, Any],
    scope: Mapping[str, Any] | None = None,
) -> SearchSchemaDefinition:
    """Read the Planner schema, with a deliberately generic legacy fallback."""

    raw_schema = plan.get("search_schema")
    if isinstance(raw_schema, Mapping):
        payload = dict(raw_schema)
        payload.setdefault("schema_id", make_id("schema", run_id, 1))
        payload.setdefault("created_by", "planner")
        try:
            return SearchSchemaDefinition.model_validate(payload)
        except ValidationError:
            # Old or malformed Planner output remains usable through the
            # compatibility shape.  The invalid payload is retained in plan
            # traces and is never silently treated as verified semantics.
            pass

    fields = _legacy_fields(plan)
    table = SearchTableDefinition(
        table_id="results",
        row_entity_type="research_target",
        description="Compatibility table until the Planner supplies a typed schema.",
        primary_key_fields=["entity_id"],
        fields=fields,
    )
    scope_data = dict(scope or {})
    contract = ScopeContract(
        target_description=normalize_text(
            str(scope_data.get("objective") or scope_data.get("question") or "")
        ),
        inclusion_rules=[str(item) for item in scope_data.get("inclusion_rules", [])],
        exclusion_rules=[str(item) for item in scope_data.get("exclusion_rules", [])],
        time_boundary=normalize_text(str(scope_data.get("date_boundary") or "")),
        geography=normalize_text(str(scope_data.get("geography") or "")),
    )
    return SearchSchemaDefinition(
        schema_id=make_id("schema", run_id, 1),
        status="provisional",
        tables=[table],
        scope_contract=contract,
        rationale=(
            "Legacy Planner output did not include a valid dynamic search_schema; "
            "this generic table preserves compatibility without inferring a domain type."
        ),
        created_by="legacy_adapter",
    )


def reconcile_schema_revision(
    previous: SearchSchemaDefinition | None,
    proposed: SearchSchemaDefinition,
    *,
    reason: str = "Planner supplied a revised run-local search schema.",
    preserved_artifact_ids: Iterable[str] = (),
) -> tuple[SearchSchemaDefinition, SchemaRevision | None]:
    """Version a semantic Planner revision without discarding prior artifacts."""

    if previous is None:
        return proposed, None
    previous_shape = previous.model_dump(
        exclude={"schema_id", "version", "status", "created_by"}
    )
    proposed_shape = proposed.model_dump(
        exclude={"schema_id", "version", "status", "created_by"}
    )
    if previous_shape == proposed_shape:
        return previous, None
    revised = proposed.model_copy(
        update={
            "schema_id": previous.schema_id,
            "version": previous.version + 1,
            "created_by": "focused_replan",
        }
    )
    changed_paths = [
        key
        for key in sorted(set(previous_shape) | set(proposed_shape))
        if previous_shape.get(key) != proposed_shape.get(key)
    ]
    revision = SchemaRevision(
        revision_id=make_id(
            "schema-revision", revised.schema_id, previous.version, revised.version
        ),
        schema_id=revised.schema_id,
        from_version=previous.version,
        to_version=revised.version,
        reason=reason,
        changed_paths=changed_paths,
        preserved_artifact_ids=list(dict.fromkeys(preserved_artifact_ids)),
    )
    return revised, revision


def source_lead_from_legacy(lead: CandidateLead | Mapping[str, Any]) -> SourceLead:
    """Convert a legacy navigation lead without promoting it to an entity."""

    parsed = lead if isinstance(lead, CandidateLead) else CandidateLead.model_validate(lead)
    return SourceLead(
        lead_id=parsed.lead_id,
        task_id=parsed.task_id,
        title=parsed.title,
        identifier=parsed.identifier,
        document_type=parsed.lead_type,
        artifact_role=parsed.artifact_role,
        target_entity_type=parsed.target_entity_type,
        origin_source_id=(parsed.origin_source_ids[0] if parsed.origin_source_ids else ""),
        anchor_quote=parsed.anchor_quote,
        extraction_strategy=parsed.extraction_strategy,
        requirement_ids=parsed.requirement_ids,
        missing_fields=parsed.missing_fields,
        status="rejected" if parsed.status == "rejected" else "unverified",
        notes=parsed.notes,
    )


def promote_entity_candidate(
    candidate: EntityCandidate,
    schema: SearchSchemaDefinition,
    *,
    known_evidence_ids: set[str],
) -> EntityCandidate | None:
    """Apply the narrow deterministic gate after the LLM chooses semantics."""

    allowed_types = {table.row_entity_type for table in schema.tables}
    if candidate.entity_type not in allowed_types:
        return None
    if candidate.scope_status != "in_scope":
        return None
    if not candidate.anchor_evidence_ids:
        return None
    if not set(candidate.anchor_evidence_ids) <= known_evidence_ids:
        return None
    return candidate.model_copy(update={"materialization_status": "materialized"})


def _node_ids(state: Mapping[str, Any], delta: ObservationDelta) -> set[str]:
    keys = (
        ("sources", "source_id"),
        ("source_leads", "lead_id"),
        ("entity_candidates", "entity_id"),
        ("evidence", "evidence_id"),
        ("claims", "claim_id"),
    )
    values = {
        str(item.get(id_key))
        for state_key, id_key in keys
        for item in state.get(state_key, [])
        if item.get(id_key)
    }
    values.update(item.source_id for item in delta.source_records)
    values.update(item.lead_id for item in delta.source_leads)
    values.update(item.entity_id for item in delta.entity_candidates)
    values.update(item.evidence_id for item in delta.evidence_records)
    values.update(item.claim_id for item in delta.claims)
    return values


def validate_observation_delta(
    state: Mapping[str, Any], delta: ObservationDelta
) -> None:
    """Validate the entire delta before any externally visible state change."""

    snapshots: dict[str, SourceSnapshot] = {
        item.source_id: item
        for item in (
            SourceSnapshot.model_validate(raw)
            for raw in [*state.get("source_snapshots", []), *[x.model_dump() for x in delta.source_snapshots]]
        )
    }
    source_ids = {
        str(item.get("source_id"))
        for item in state.get("sources", [])
        if item.get("source_id")
    } | {item.source_id for item in delta.source_records}
    evidence_ids = {
        str(item.get("evidence_id"))
        for item in state.get("evidence", [])
        if item.get("evidence_id")
    } | {item.evidence_id for item in delta.evidence_records}
    entity_ids = {
        str(item.get("entity_id"))
        for item in state.get("entity_candidates", [])
        if item.get("entity_id")
    } | {item.entity_id for item in delta.entity_candidates}
    claim_ids = {
        str(item.get("claim_id"))
        for item in state.get("claims", [])
        if item.get("claim_id")
    } | {item.claim_id for item in delta.claims}

    for evidence in delta.evidence_records:
        if evidence.source_id not in source_ids:
            raise ObservationValidationError(
                f"unknown_evidence_source:{evidence.evidence_id}:{evidence.source_id}"
            )
        snapshot = snapshots.get(evidence.source_id)
        if snapshot is None:
            raise ObservationValidationError(
                f"missing_source_snapshot:{evidence.evidence_id}:{evidence.source_id}"
            )
        if normalize_text(evidence.quote) not in normalize_text(snapshot.content):
            raise ObservationValidationError(
                f"quote_not_in_snapshot:{evidence.evidence_id}"
            )

    for binding in delta.bindings:
        if binding.evidence_id not in evidence_ids:
            raise ObservationValidationError(
                f"unknown_binding_evidence:{binding.binding_id}:{binding.evidence_id}"
            )
        if binding.entity_id not in entity_ids:
            raise ObservationValidationError(
                f"unknown_binding_entity:{binding.binding_id}:{binding.entity_id}"
            )
        if not binding.field_key:
            raise ObservationValidationError(f"empty_binding_field:{binding.binding_id}")

    for link in delta.links:
        if link.claim_id not in claim_ids:
            raise ObservationValidationError(
                f"unknown_link_claim:{link.link_id}:{link.claim_id}"
            )
        if link.evidence_id not in evidence_ids:
            raise ObservationValidationError(
                f"unknown_link_evidence:{link.link_id}:{link.evidence_id}"
            )

    nodes = _node_ids(state, delta)
    for edge in delta.graph_edges:
        if edge.source_node_id not in nodes or edge.target_node_id not in nodes:
            raise ObservationValidationError(f"dangling_evidence_edge:{edge.edge_id}")
        if edge.evidence_id and edge.evidence_id not in evidence_ids:
            raise ObservationValidationError(f"unknown_edge_evidence:{edge.edge_id}")


def prune_invalid_observation_items(
    state: Mapping[str, Any], delta: ObservationDelta
) -> tuple[ObservationDelta, dict[str, Any]]:
    """Remove invalid semantic items without quarantining the whole Worker delta.

    Evidence is validated independently.  Dependent links, bindings, graph edges,
    and newly unsupported claims are removed, while unrelated observations from
    the same Worker remain commit-eligible.
    """

    snapshots: dict[str, SourceSnapshot] = {
        item.source_id: item
        for item in (
            SourceSnapshot.model_validate(raw)
            for raw in [
                *state.get("source_snapshots", []),
                *[item.model_dump(mode="json") for item in delta.source_snapshots],
            ]
        )
    }
    source_ids = {
        str(item.get("source_id"))
        for item in state.get("sources", [])
        if item.get("source_id")
    } | {item.source_id for item in delta.source_records}
    existing_evidence_ids = {
        str(item.get("evidence_id"))
        for item in state.get("evidence", [])
        if item.get("evidence_id")
    }
    existing_claim_ids = {
        str(item.get("claim_id"))
        for item in state.get("claims", [])
        if item.get("claim_id")
    }
    existing_entity_ids = {
        str(item.get("entity_id"))
        for item in state.get("entity_candidates", [])
        if item.get("entity_id")
    }

    rejected_evidence: dict[str, str] = {}
    valid_evidence: list[EvidenceRecord] = []
    for evidence in delta.evidence_records:
        reason = ""
        if evidence.source_id not in source_ids:
            reason = f"unknown_evidence_source:{evidence.source_id}"
        else:
            snapshot = snapshots.get(evidence.source_id)
            if snapshot is None:
                reason = f"missing_source_snapshot:{evidence.source_id}"
            elif normalize_text(evidence.quote) not in normalize_text(snapshot.content):
                reason = "quote_not_in_snapshot"
        if reason:
            rejected_evidence[evidence.evidence_id] = reason
        else:
            valid_evidence.append(evidence)

    available_evidence_ids = existing_evidence_ids | {
        item.evidence_id for item in valid_evidence
    }
    valid_entities: list[EntityCandidate] = []
    rejected_entity_ids: list[str] = []
    for entity in delta.entity_candidates:
        anchors = [
            evidence_id
            for evidence_id in entity.anchor_evidence_ids
            if evidence_id in available_evidence_ids
        ]
        if entity.anchor_evidence_ids and not anchors and entity.entity_id not in existing_entity_ids:
            rejected_entity_ids.append(entity.entity_id)
            continue
        valid_entities.append(entity.model_copy(update={"anchor_evidence_ids": anchors}))

    available_entity_ids = existing_entity_ids | {
        item.entity_id for item in valid_entities
    }
    potential_claim_ids = existing_claim_ids | {item.claim_id for item in delta.claims}
    valid_links = [
        link
        for link in delta.links
        if link.evidence_id in available_evidence_ids
        and link.claim_id in potential_claim_ids
    ]
    existing_supported_claim_ids = {
        str(item.get("claim_id"))
        for item in state.get("links", [])
        if item.get("claim_id")
        and str(item.get("evidence_id")) in available_evidence_ids
    }
    incoming_supported_claim_ids = {link.claim_id for link in valid_links}
    originally_linked_claim_ids = {link.claim_id for link in delta.links}
    valid_claims = [
        claim
        for claim in delta.claims
        if claim.claim_id in existing_claim_ids
        or claim.claim_id not in originally_linked_claim_ids
        or claim.claim_id in existing_supported_claim_ids
        or claim.claim_id in incoming_supported_claim_ids
    ]
    available_claim_ids = existing_claim_ids | {item.claim_id for item in valid_claims}
    valid_links = [link for link in valid_links if link.claim_id in available_claim_ids]

    valid_bindings = [
        binding
        for binding in delta.bindings
        if binding.evidence_id in available_evidence_ids
        and binding.entity_id in available_entity_ids
        and bool(binding.field_key)
    ]
    valid_binding_ids = {item.binding_id for item in valid_bindings}

    provisional = delta.model_copy(
        update={
            "entity_candidates": valid_entities,
            "evidence_records": valid_evidence,
            "claims": valid_claims,
        }
    )
    nodes = _node_ids(state, provisional)
    valid_edges = [
        edge
        for edge in delta.graph_edges
        if edge.source_node_id in nodes
        and edge.target_node_id in nodes
        and (not edge.evidence_id or edge.evidence_id in available_evidence_ids)
    ]
    valid_coverage_changes = [
        change
        for change in delta.coverage_changes
        if (not change.entity_id or change.entity_id in available_entity_ids)
        and (not change.binding_id or change.binding_id in valid_binding_ids)
    ]

    pruned = delta.model_copy(
        update={
            "entity_candidates": valid_entities,
            "evidence_records": valid_evidence,
            "claims": valid_claims,
            "links": valid_links,
            "bindings": valid_bindings,
            "graph_edges": valid_edges,
            "coverage_changes": valid_coverage_changes,
        }
    )
    report = {
        "rejected_evidence": [
            {"evidence_id": evidence_id, "reason": reason}
            for evidence_id, reason in rejected_evidence.items()
        ],
        "rejected_claim_ids": sorted(
            {item.claim_id for item in delta.claims}
            - {item.claim_id for item in valid_claims}
        ),
        "rejected_link_ids": sorted(
            {item.link_id for item in delta.links}
            - {item.link_id for item in valid_links}
        ),
        "rejected_binding_ids": sorted(
            {item.binding_id for item in delta.bindings}
            - {item.binding_id for item in valid_bindings}
        ),
        "rejected_edge_ids": sorted(
            {item.edge_id for item in delta.graph_edges}
            - {item.edge_id for item in valid_edges}
        ),
        "rejected_entity_ids": sorted(set(rejected_entity_ids)),
        "preserved_claim_ids": sorted(item.claim_id for item in valid_claims),
    }
    return pruned, report


def calculate_dual_coverage(state: Mapping[str, Any]) -> tuple[CellCoverage, ScopeCompleteness]:
    """Calculate cells conservatively and never infer scope completion from them."""

    schemas = state.get("search_schemas", [])
    schema = SearchSchemaDefinition.model_validate(schemas[-1]) if schemas else None
    entities = [
        EntityCandidate.model_validate(item)
        for item in state.get("entity_candidates", [])
        if item.get("materialization_status") == "materialized"
        and item.get("scope_status") == "in_scope"
    ]
    field_keys_by_type: dict[str, list[str]] = {}
    if schema is not None:
        field_keys_by_type = {
            table.row_entity_type: [field.field_key for field in table.fields if field.required]
            for table in schema.tables
        }
    materialized_entity_ids = {entity.entity_id for entity in entities}
    expected_cells: set[tuple[str, str]] = set()
    for entity in entities:
        expected_cells.update(
            (entity.entity_id, field_key)
            for field_key in field_keys_by_type.get(entity.entity_type, [])
        )

    bindings = [
        EvidenceBinding.model_validate(item)
        for item in state.get("evidence_bindings", [])
        if item.get("status") != "rejected"
        and str(item.get("entity_id")) in materialized_entity_ids
    ]
    if not expected_cells and bindings:
        # A provisional schema with no declared fields may still expose
        # observed cells, but only for materialized in-scope entities.
        expected_cells.update((item.entity_id, item.field_key) for item in bindings)
    valid_fields = {
        (entity.entity_id, field_key)
        for entity in entities
        for field_key in field_keys_by_type.get(entity.entity_type, [])
    }
    binding_status: dict[tuple[str, str], str] = {
        key: "missing" for key in expected_cells
    }
    for binding in bindings:
        key = (binding.entity_id, binding.field_key)
        if valid_fields and key not in valid_fields:
            continue
        current = binding_status.get(key, "missing")
        if binding.status == "conflicting":
            binding_status[key] = "uncertain"
        elif binding.status == "supported" and current not in {"uncertain", "unreachable"}:
            binding_status[key] = "filled"
        elif current == "missing":
            binding_status[key] = "uncertain"

    # Explicit coverage changes can mark a reachable cell as uncertain or
    # unreachable without fabricating a filled value. They never create a
    # cell for an unknown/non-materialized entity.
    for raw_change in state.get("coverage_changes", []):
        key = (str(raw_change.get("entity_id") or ""), str(raw_change.get("field_key") or ""))
        if key not in expected_cells:
            continue
        new_status = str(raw_change.get("new_status") or "")
        if new_status == "unreachable" and binding_status.get(key) != "filled":
            binding_status[key] = "unreachable"
        elif new_status == "uncertain" and binding_status.get(key) not in {"filled", "unreachable"}:
            binding_status[key] = "uncertain"
        elif new_status == "not_applicable" and binding_status.get(key) == "missing":
            binding_status[key] = "not_applicable"

    total = len(expected_cells)
    filled = sum(value == "filled" for value in binding_status.values())
    uncertain = sum(value == "uncertain" for value in binding_status.values())
    unreachable = sum(value == "unreachable" for value in binding_status.values())
    not_applicable = sum(value == "not_applicable" for value in binding_status.values())
    missing = max(0, total - filled - uncertain - unreachable - not_applicable)
    cell = CellCoverage(
        known_entity_count=len(entities),
        materialized_cell_count=total,
        filled_cell_count=filled,
        uncertain_cell_count=uncertain,
        unreachable_cell_count=unreachable,
        not_applicable_cell_count=not_applicable,
        missing_cell_count=missing,
        ratio=(filled / total if total else 0.0),
    )

    previous = state.get("scope_completeness") or {}
    containers = [
        str(item.get("lead_id"))
        for item in state.get("source_leads", [])
        if item.get("artifact_role") == "entity_container"
        and item.get("status") not in {"rejected"}
    ]
    prior_status = str(previous.get("status") or "unknown")
    if prior_status == "unknown" and (entities or containers):
        prior_status = "open"
    prior_unresolved = [
        str(item) for item in previous.get("unresolved_container_ids", []) if str(item)
    ]
    if prior_status in {"unknown", "open", "stabilizing"}:
        unresolved_container_ids = list(dict.fromkeys([*prior_unresolved, *containers]))
    else:
        # Once the controller/LLM has explicitly audited scope, do not reopen
        # every historical container merely because it remains in the ledger.
        unresolved_container_ids = prior_unresolved
    scope = ScopeCompleteness.model_validate(
        {
            **previous,
            "status": prior_status,
            "discovered_entity_count": len(
                [
                    item
                    for item in state.get("entity_candidates", [])
                    if item.get("scope_status") != "out_of_scope"
                ]
            ),
            "unresolved_container_ids": unresolved_container_ids,
        }
    )
    return cell, scope


def commit_observation_delta(
    state: Mapping[str, Any],
    delta: ObservationDelta,
    *,
    append_only_evidence: bool = False,
) -> dict[str, Any]:
    """Return a complete new search-state slice or an idempotent no-op."""

    committed = {
        str(item.get("delta_id"))
        for item in state.get("observation_commits", [])
        if item.get("delta_id")
    }
    if delta.delta_id in committed:
        return {
            "already_committed": True,
            "delta_id": delta.delta_id,
            "state_patch": {},
        }

    validate_observation_delta(state, delta)
    merge_records = _append_only_by_id if append_only_evidence else _merge_by_id
    incoming_edges = [item.model_dump(mode="json") for item in delta.graph_edges]
    if append_only_evidence:
        existing_bindings = [
            EvidenceBinding.model_validate(item)
            for item in state.get("evidence_bindings", [])
        ]
        for incoming in delta.bindings:
            for current in existing_bindings:
                if (
                    incoming.entity_id == current.entity_id
                    and incoming.field_key == current.field_key
                    and normalize_text(str(incoming.proposed_value or "")).casefold()
                    != normalize_text(str(current.proposed_value or "")).casefold()
                    and incoming.proposed_value is not None
                    and current.proposed_value is not None
                ):
                    incoming_edges.append(
                        EvidenceEdge(
                            edge_id=make_id(
                                "edge-conflict",
                                current.evidence_id,
                                incoming.evidence_id,
                                incoming.entity_id,
                                incoming.field_key,
                            ),
                            source_node_id=current.evidence_id,
                            target_node_id=incoming.evidence_id,
                            relation="conflicts",
                            evidence_id=incoming.evidence_id,
                            rationale=(
                                "Append-only evidence graph preserved distinct values "
                                "for the same entity field."
                            ),
                        ).model_dump(mode="json")
                    )
    patch: dict[str, Any] = {
        "sources": _merge_by_id(
            state.get("sources", []),
            [item.model_dump(mode="json") for item in delta.source_records],
            "source_id",
        ),
        "source_snapshots": _merge_by_id(
            state.get("source_snapshots", []),
            [item.model_dump(mode="json") for item in delta.source_snapshots],
            "source_id",
        ),
        "source_leads": _merge_by_id(
            state.get("source_leads", []),
            [item.model_dump(mode="json") for item in delta.source_leads],
            "lead_id",
        ),
        "entity_candidates": _merge_by_id(
            state.get("entity_candidates", []),
            [item.model_dump(mode="json") for item in delta.entity_candidates],
            "entity_id",
        ),
        "evidence": merge_records(
            state.get("evidence", []),
            [item.model_dump(mode="json") for item in delta.evidence_records],
            "evidence_id",
        ),
        "claims": merge_records(
            state.get("claims", []),
            [item.model_dump(mode="json") for item in delta.claims],
            "claim_id",
        ),
        "links": merge_records(
            state.get("links", []),
            [item.model_dump(mode="json") for item in delta.links],
            "link_id",
        ),
        "evidence_bindings": merge_records(
            state.get("evidence_bindings", []),
            [item.model_dump(mode="json") for item in delta.bindings],
            "binding_id",
        ),
        "evidence_edges": merge_records(
            state.get("evidence_edges", []),
            incoming_edges,
            "edge_id",
        ),
        "failure_memory": _merge_by_id(
            state.get("failure_memory", []),
            [item.model_dump(mode="json") for item in delta.failure_records],
            "failure_id",
        ),
        "coverage_changes": _merge_by_id(
            state.get("coverage_changes", []),
            [item.model_dump(mode="json") for item in delta.coverage_changes],
            "change_id",
        ),
        "observation_commits": [
            *state.get("observation_commits", []),
            {
                "delta_id": delta.delta_id,
                "task_id": delta.task_id,
                "created_at": delta.created_at,
            },
        ],
    }
    cell, scope = calculate_dual_coverage({**state, **patch})
    patch["cell_coverage"] = cell.model_dump(mode="json")
    patch["scope_completeness"] = scope.model_dump(mode="json")
    return {
        "already_committed": False,
        "delta_id": delta.delta_id,
        "state_patch": patch,
    }


def build_binding_edges(bindings: Iterable[EvidenceBinding]) -> list[EvidenceEdge]:
    """Create typed evidence-to-field edges from already validated bindings."""

    edges: list[EvidenceEdge] = []
    for binding in bindings:
        relation = "conflicts" if binding.status == "conflicting" else "supports"
        edges.append(
            EvidenceEdge(
                edge_id=make_id("edge", binding.binding_id, relation),
                source_node_id=binding.evidence_id,
                target_node_id=binding.entity_id,
                relation=relation,
                evidence_id=binding.evidence_id,
                rationale=f"Field binding: {binding.field_key}",
            )
        )
    return edges


def schema_field_keys(schema: SearchSchemaDefinition) -> set[str]:
    return {field.field_key for table in schema.tables for field in table.fields}


def schema_relations(schema: SearchSchemaDefinition) -> list[SearchRelationDefinition]:
    return list(schema.relations)
