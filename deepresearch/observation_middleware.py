"""Fetch-boundary conversion from model observations to typed state deltas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .failure_memory import (
    failures_from_extraction_traces,
    failures_from_fetch_traces,
    failures_from_search_traces,
)
from .ids import make_id, normalize_text
from .schemas import (
    ClaimEvidenceLink,
    ClaimRecord,
    EntityCandidate,
    EvidenceBinding,
    EvidenceEdge,
    EvidenceExtractionResult,
    EvidenceRecord,
    FailureRecord,
    ObservationDelta,
    ResearchTask,
    SearchSchemaDefinition,
    SourceLead,
    SourceRecord,
    SourceSnapshot,
)
from .search_state import build_binding_edges, promote_entity_candidate, source_lead_from_legacy


def observations_from_extraction(
    *,
    run_id: str,
    task: ResearchTask,
    source: SourceRecord,
    parsed: EvidenceExtractionResult,
    accepted_candidates: Sequence[Mapping[str, Any]],
    evidence: Sequence[EvidenceRecord],
    schema: SearchSchemaDefinition | None,
) -> dict[str, list[dict[str, Any]]]:
    """Preserve LLM semantic choices while anchoring them to accepted quotes."""

    source_leads: list[SourceLead] = []
    entities: dict[str, EntityCandidate] = {}
    bindings: list[EvidenceBinding] = []
    edges: list[EvidenceEdge] = []

    if parsed.artifact_role != "unknown":
        anchor_quote = evidence[0].quote if evidence else ""
        source_leads.append(
            SourceLead(
                lead_id=make_id("source-lead", task.task_id, source.source_id),
                task_id=task.task_id,
                title=source.title,
                identifier=source.canonical_uri,
                document_type=source.mime_type or "web",
                artifact_role=parsed.artifact_role,
                target_entity_type=parsed.target_entity_type,
                origin_source_id=source.source_id,
                anchor_quote=anchor_quote,
                extraction_strategy=parsed.extraction_strategy,
                requirement_ids=task.requirement_ids,
                status="usable" if evidence else "unverified",
            )
        )

    for index, (candidate, evidence_item) in enumerate(
        zip(accepted_candidates, evidence, strict=False)
    ):
        entity_id = normalize_text(str(candidate.get("entity_id") or ""))
        entity_label = normalize_text(str(candidate.get("entity_label") or ""))
        entity_type = normalize_text(str(candidate.get("entity_type") or ""))
        if entity_id and entity_label and entity_type:
            current = EntityCandidate(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_label=entity_label,
                aliases=[
                    normalize_text(str(item))
                    for item in candidate.get("entity_aliases", [])
                    if normalize_text(str(item))
                ],
                identifiers=[
                    normalize_text(str(item))
                    for item in candidate.get("entity_identifiers", [])
                    if normalize_text(str(item))
                ],
                origin_source_ids=[source.source_id],
                anchor_evidence_ids=[evidence_item.evidence_id],
                scope_status=str(candidate.get("entity_scope_status") or "uncertain"),
                rationale="Entity semantics supplied by the extraction model.",
            )
            existing = entities.get(entity_id)
            if existing is not None:
                current = current.model_copy(
                    update={
                        "aliases": list(dict.fromkeys([*existing.aliases, *current.aliases])),
                        "identifiers": list(
                            dict.fromkeys([*existing.identifiers, *current.identifiers])
                        ),
                        "anchor_evidence_ids": list(
                            dict.fromkeys(
                                [*existing.anchor_evidence_ids, *current.anchor_evidence_ids]
                            )
                        ),
                    }
                )
            entities[entity_id] = current
            if parsed.artifact_role == "entity_container":
                edges.append(
                    EvidenceEdge(
                        edge_id=make_id(
                            "edge", source.source_id, entity_id, "contains", index
                        ),
                        source_node_id=source.source_id,
                        target_node_id=entity_id,
                        relation="contains",
                        evidence_id=evidence_item.evidence_id,
                        rationale="The extraction model identified this page as an entity container.",
                    )
                )
            else:
                edges.append(
                    EvidenceEdge(
                        edge_id=make_id(
                            "edge", source.source_id, entity_id, "mentions", index
                        ),
                        source_node_id=source.source_id,
                        target_node_id=entity_id,
                        relation="mentions",
                        evidence_id=evidence_item.evidence_id,
                    )
                )

        field_key = normalize_text(str(candidate.get("field_key") or ""))
        recovered_task_binding = False
        recovered_table_id = ""
        candidate_requirement_ids = {
            str(item)
            for item in candidate.get("requirement_ids", [])
            if str(item)
        }
        task_requirement_ids = {
            *[str(item) for item in task.requirement_ids if str(item)],
            *[
                str(item)
                for item in task.produces_requirement_ids
                if str(item)
            ],
        }
        evidence_requirement_ids = {
            str(item) for item in evidence_item.requirement_ids if str(item)
        }
        # Recover only the fully unambiguous case. The task assignment is a
        # trusted run-local contract, while the extraction model may omit
        # redundant entity/field metadata. Never guess when either side has
        # multiple targets, the claim/evidence belongs to another requirement,
        # or the field does not resolve to exactly one schema table.
        if (
            not entity_id
            and not field_key
            and len(task.target_entity_ids) == 1
            and len(task.target_field_keys) == 1
            and candidate_requirement_ids & task_requirement_ids
            and evidence_requirement_ids & task_requirement_ids
            and schema is not None
        ):
            target_field = normalize_text(task.target_field_keys[0])
            matching_tables = [
                table
                for table in schema.tables
                if any(field.field_key == target_field for field in table.fields)
                and (
                    not parsed.target_entity_type
                    or table.row_entity_type == parsed.target_entity_type
                )
            ]
            if len(matching_tables) == 1:
                entity_id = normalize_text(task.target_entity_ids[0])
                field_key = target_field
                recovered_table_id = matching_tables[0].table_id
                recovered_task_binding = True
        if entity_id and field_key:
            requirement_id = next(
                (
                    str(item)
                    for item in candidate.get("requirement_ids", [])
                    if str(item) in task.requirement_ids
                    or str(item) in task.produces_requirement_ids
                ),
                "",
            )
            binding = EvidenceBinding(
                binding_id=make_id(
                    "binding", entity_id, field_key, evidence_item.evidence_id
                ),
                evidence_id=evidence_item.evidence_id,
                table_id=recovered_table_id,
                entity_id=entity_id,
                field_key=field_key,
                requirement_id=requirement_id,
                proposed_value=candidate.get("field_value"),
                status=str(candidate.get("binding_status") or "supported"),
                rationale=(
                    "Entity and field recovered from an unambiguous task target; "
                    "the claim and evidence match the task requirement."
                    if recovered_task_binding
                    else "Field semantics supplied by the extraction model."
                ),
            )
            bindings.append(binding)

    known_evidence_ids = {item.evidence_id for item in evidence}
    if schema is not None:
        entities = {
            entity_id: promoted or candidate
            for entity_id, candidate in entities.items()
            for promoted in [
                promote_entity_candidate(
                    candidate, schema, known_evidence_ids=known_evidence_ids
                )
            ]
        }
    edges.extend(build_binding_edges(bindings))
    return {
        "source_leads": [item.model_dump(mode="json") for item in source_leads],
        "entity_candidates": [
            item.model_dump(mode="json") for item in entities.values()
        ],
        "evidence_bindings": [item.model_dump(mode="json") for item in bindings],
        "evidence_edges": [item.model_dump(mode="json") for item in edges],
    }


def delta_from_worker_result(
    *,
    run_id: str,
    task: ResearchTask,
    result: Mapping[str, Any],
    existing_failure_records: Iterable[FailureRecord | Mapping[str, Any]] = (),
    enhanced_failure_memory: bool = False,
) -> ObservationDelta:
    """Build one idempotent delta from every artifact returned by a Worker."""

    source_leads = [
        SourceLead.model_validate(item) for item in result.get("source_leads", [])
    ]
    known_lead_ids = {item.lead_id for item in source_leads}
    for raw in result.get("candidate_leads", []):
        converted = source_lead_from_legacy(raw)
        if converted.lead_id not in known_lead_ids:
            source_leads.append(converted)
            known_lead_ids.add(converted.lead_id)

    failures = [
        FailureRecord.model_validate(item) for item in result.get("failure_memory", [])
    ]
    failures.extend(
        failures_from_fetch_traces(
            run_id=run_id,
            task_id=task.task_id,
            traces=result.get("fetch_traces", []),
            existing=existing_failure_records,
            enhanced=enhanced_failure_memory,
        )
    )
    if enhanced_failure_memory:
        failures.extend(
            failures_from_search_traces(
                run_id=run_id,
                task_id=task.task_id,
                traces=result.get("search_traces", []),
                existing=[*existing_failure_records, *failures],
            )
        )
        failures.extend(
            failures_from_extraction_traces(
                task_id=task.task_id,
                traces=result.get("extraction_traces", []),
                existing=[*existing_failure_records, *failures],
            )
        )
    artifact_ids = sorted(
        [str(item.get("source_id")) for item in result.get("sources", [])]
        + [str(item.get("evidence_id")) for item in result.get("evidence", [])]
        + [str(item.get("entity_id")) for item in result.get("entity_candidates", [])]
        + [str(item.get("binding_id")) for item in result.get("evidence_bindings", [])]
    )
    return ObservationDelta(
        delta_id=make_id("delta", run_id, task.task_id, task.attempt, artifact_ids),
        task_id=task.task_id,
        source_records=[SourceRecord.model_validate(item) for item in result.get("sources", [])],
        source_snapshots=[
            SourceSnapshot.model_validate(item)
            for item in result.get("source_snapshots", [])
        ],
        source_leads=source_leads,
        entity_candidates=[
            EntityCandidate.model_validate(item)
            for item in result.get("entity_candidates", [])
        ],
        evidence_records=[
            EvidenceRecord.model_validate(item) for item in result.get("evidence", [])
        ],
        claims=[ClaimRecord.model_validate(item) for item in result.get("claims", [])],
        links=[
            ClaimEvidenceLink.model_validate(item) for item in result.get("links", [])
        ],
        bindings=[
            EvidenceBinding.model_validate(item)
            for item in result.get("evidence_bindings", [])
        ],
        graph_edges=[
            EvidenceEdge.model_validate(item) for item in result.get("evidence_edges", [])
        ],
        failure_records=failures,
    )
