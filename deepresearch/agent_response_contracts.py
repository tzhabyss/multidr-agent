"""Normalization and validation contracts for agent JSON responses.

This module is intentionally independent from the agent implementations.  It
keeps provider-recovery validation reusable without importing the  Planner,
Worker, or Synthesizer orchestration code.
"""

from __future__ import annotations

from typing import Any

from .schemas import WriterDraft, WriterReview


def normalize_writer_draft_payload(value: Any) -> dict[str, Any]:
    """Promote heading-like paragraph objects emitted by the Writer."""

    if not isinstance(value, dict):
        return {}

    normalized = dict(value)
    sections: list[dict[str, Any]] = []

    def paragraph_from(
        item: Any, inherited: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        inherited = inherited or {}
        own_claim_ids = item.get("claim_ids") or inherited.get("claim_ids") or []
        own_requirement_ids = (
            item.get("requirement_ids") or inherited.get("requirement_ids") or []
        )
        if isinstance(item.get("text"), str) and item["text"].strip():
            paragraph = dict(item)
            paragraph["claim_ids"] = own_claim_ids
            paragraph["requirement_ids"] = own_requirement_ids
            paragraph.pop("paragraphs", None)
            paragraph.pop("tables", None)
            return paragraph
        return None

    def promote_section(raw_section: dict[str, Any]) -> list[dict[str, Any]]:
        section = dict(raw_section)
        direct_paragraphs: list[dict[str, Any]] = []
        child_sections: list[dict[str, Any]] = []
        for item in raw_section.get("paragraphs", []):
            if not isinstance(item, dict):
                continue
            is_nested_section = bool(
                isinstance(item.get("title"), str)
                and item["title"].strip()
                and (isinstance(item.get("paragraphs"), list) or item.get("tables"))
            )
            if is_nested_section:
                child_sections.extend(promote_section(item))
                continue
            paragraph = paragraph_from(item)
            if paragraph is not None:
                direct_paragraphs.append(paragraph)
            for child in item.get("paragraphs", []):
                child_paragraph = paragraph_from(child, item)
                if child_paragraph is not None:
                    direct_paragraphs.append(child_paragraph)
        section["title"] = str(raw_section.get("title") or "").strip()
        section["paragraphs"] = direct_paragraphs
        section["tables"] = (
            list(raw_section.get("tables", []))
            if isinstance(raw_section.get("tables"), list)
            else []
        )
        return [section, *child_sections]

    for raw_section in value.get("sections", []):
        if isinstance(raw_section, dict):
            sections.extend(promote_section(raw_section))
    normalized["sections"] = sections
    return normalized


def validate_planner_payload(value: dict[str, Any]) -> None:
    """Coarsely validate hedged Planner responses."""

    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Planner response must contain a tasks list")
    if not tasks:
        raise ValueError("Planner response contains no usable tasks")
    if any(not isinstance(item, dict) for item in tasks):
        raise ValueError("Planner tasks must be JSON objects")


def validate_writer_payload(value: dict[str, Any]) -> None:
    draft = WriterDraft.model_validate(normalize_writer_draft_payload(value))
    has_text = any(
        paragraph.text.strip()
        for section in draft.sections
        for paragraph in section.paragraphs
    )
    has_table_text = any(
        cell.text.strip()
        for section in draft.sections
        for table in section.tables
        for row in table.rows
        for cell in row
    )
    if not (has_text or has_table_text or draft.missing_dimensions):
        raise ValueError("Writer response contains no usable report content")


def validate_review_payload(value: dict[str, Any]) -> None:
    WriterReview.model_validate(value)

