from __future__ import annotations

from deepresearch.schemas import (
    EvidenceExtractionResult,
    FetchedDocument,
    ResearchTask,
    SearchFieldDefinition,
    SearchSchemaDefinition,
    SearchTableDefinition,
    ScopeContract,
    TaskBudget,
)
from tools.run_live_task75_extraction_breakpoint import analyze_extraction_pipeline


def test_breakpoint_detects_entity_metadata_loss_before_observation_delta():
    document = FetchedDocument(
        requested_url="https://example.test/ssass",
        final_url="https://example.test/ssass",
        title="SSaSS",
        content="The Salt Substitute and Stroke Study (SSaSS) included 20,995 participants.",
        content_type="text/plain",
        http_status=200,
    )
    task = ResearchTask(
        task_id="task-entity",
        objective="Extract eligible trials.",
        research_dimension="trial inventory",
        research_object="salt substitute trials",
        task_kind="gather",
        queries=[],
        requirement_ids=["req-1"],
        budget=TaskBudget(
            max_searches=1,
            max_fetches=1,
            max_wall_time_seconds=60,
        ),
    )
    schema = SearchSchemaDefinition(
        schema_id="schema-1",
        tables=[
            SearchTableDefinition(
                table_id="trials",
                row_entity_type="trial",
                fields=[
                    SearchFieldDefinition(
                        field_key="trial_identity",
                        label="Trial",
                    )
                ],
            )
        ],
        scope_contract=ScopeContract(target_description="eligible trials"),
    )
    parsed = EvidenceExtractionResult.model_validate(
        {
            "artifact_role": "target_entity",
            "target_entity_type": "trial",
            "claims": [
                {
                    "text": "SSaSS included 20,995 participants.",
                    "quote": "The Salt Substitute and Stroke Study (SSaSS) included 20,995 participants.",
                    "requirement_ids": ["req-1"],
                    "entity_id": "entity-ssass",
                    "entity_label": "SSaSS",
                    "entity_type": "trial",
                    "entity_scope_status": "in_scope",
                    "field_key": "trial_identity",
                    "field_value": "SSaSS",
                }
            ],
        }
    )

    result = analyze_extraction_pipeline(
        run_id="run-1",
        task=task,
        document=document,
        parsed=parsed,
        schema=schema,
    )

    assert result["status"] == "fail"
    assert result["breakpoint"] == "filter_extracted_claims_drops_entity_metadata"
    assert result["observed"]["raw_entity_claim_count"] == 1
    assert result["observed"]["accepted_claim_count"] == 1
    assert result["observed"]["filtered_entity_claim_count"] == 0
    assert result["observed"]["typed_entity_candidate_count"] == 0
    assert result["observed"]["committed_entity_count"] == 0
