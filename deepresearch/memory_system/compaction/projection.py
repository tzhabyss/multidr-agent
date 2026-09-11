"""Query-aware, role-aware ephemeral projections over durable memories."""

from __future__ import annotations

import re
from typing import Any

from .controller import CompactionController, estimate_tokens
from .models import ProjectionResult


_SENTENCE_PATTERN = re.compile(r"[^。！？!?\n]+[。！？!?]?", re.MULTILINE)
_ALLOWED_KEYS = {
    "memory_id",
    "episode_id",
    "run_id",
    "session_id",
    "memory_type",
    "origin",
    "research_brief",
    "status",
    "summary",
    "abstract",
    "category",
    "scope",
    "revision",
    "confidence",
    "entities",
    "user_corrections",
    "preferences",
    "decisions",
    "constraints",
    "open_gaps",
    "successful_queries",
    "failed_queries",
    "report_path",
    "retrieval_score",
    "retrieval_channels",
    "must_keep_spans",
    "detail_ref",
    "temporal_status",
    "state_authority",
    "state_projection_allowed",
    "historical_state_fields",
}


def terms(value: str) -> set[str]:
    lowered = value.casefold()
    result = set(re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    result.update(
        chinese[index : index + 2]
        for index in range(max(0, len(chinese) - 1))
    )
    return {item for item in result if item}


def _memory_id(item: dict[str, Any]) -> str:
    return str(
        item.get("memory_id")
        or item.get("episode_id")
        or item.get("document_id")
        or ""
    )


def _split_sentences(value: str) -> list[str]:
    return [
        " ".join(match.group(0).split()).strip()
        for match in _SENTENCE_PATTERN.finditer(value)
        if match.group(0).strip()
    ]


def _units(item: dict[str, Any]) -> list[tuple[str, str, bool]]:
    result: list[tuple[str, str, bool]] = []
    protected_fact = str(item.get("category") or "") in {
        "preference",
        "constraint",
        "correction",
        "project_decision",
    }
    for field in ("abstract", "summary", "research_brief"):
        value = str(item.get(field) or "").strip()
        result.extend(
            (text, field, protected_fact and field == "summary")
            for text in _split_sentences(value)
        )
    for field in (
        "constraints",
        "decisions",
        "user_corrections",
        "open_gaps",
        "successful_queries",
        "failed_queries",
    ):
        values = item.get(field, [])
        if isinstance(values, list):
            result.extend(
                (
                    " ".join(str(value).split()).strip(),
                    field,
                    field in {"constraints", "decisions", "user_corrections"},
                )
                for value in values
                if str(value).strip()
            )
    spans = item.get("must_keep_spans", [])
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict) and str(span.get("text") or "").strip():
                result.append((str(span["text"]).strip(), "must_keep", True))
    deduped: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for text, field, protected in result:
        identity = " ".join(text.casefold().split())
        if not identity or identity in seen:
            continue
        seen.add(identity)
        deduped.append((text, field, protected))
    return deduped


def _role_boost(role: str, field: str) -> float:
    preferred = {
        "clarifier": {"constraints", "user_corrections", "decisions", "must_keep"},
        "planner": {
            "research_brief",
            "constraints",
            "decisions",
            "open_gaps",
            "failed_queries",
            "must_keep",
        },
        "worker": {
            "research_brief",
            "open_gaps",
            "failed_queries",
            "summary",
        },
        "synthesizer": {
            "summary",
            "constraints",
            "decisions",
            "open_gaps",
            "must_keep",
        },
    }
    return 0.15 if field in preferred.get(role, set()) else 0.0


def project_memories(
    items: list[dict[str, Any]],
    *,
    query: str,
    role: str,
    budget_tokens: int,
    max_items: int,
    sentence_limit: int,
    ratios: tuple[float, float, float, float] = (0.60, 0.75, 0.88, 0.95),
) -> ProjectionResult:
    """Create a bounded temporary view; source memories remain untouched."""

    raw_tokens = sum(estimate_tokens(str(item)) for item in items)
    level = CompactionController.projection_level(
        raw_tokens,
        budget_tokens,
        light_ratio=ratios[0],
        query_ratio=ratios[1],
        summary_ratio=ratios[2],
        emergency_ratio=ratios[3],
    )
    effective_max_items = max_items
    effective_sentence_limit = sentence_limit
    if level == "query_aware":
        effective_sentence_limit = max(3, sentence_limit - 2)
    elif level == "summarize":
        effective_max_items = max(1, min(max_items, max_items // 2 + 1))
        effective_sentence_limit = max(2, sentence_limit // 2)
    elif level == "emergency":
        effective_max_items = min(max_items, 3)
        effective_sentence_limit = 2
    query_terms = terms(query)
    ranked_items: list[tuple[float, dict[str, Any], list[tuple[float, str, str, bool]]]] = []
    for rank, item in enumerate(items):
        base = float(item.get("retrieval_score") or 0.0)
        ranked_units: list[tuple[float, str, str, bool]] = []
        for text, field, protected in _units(item):
            unit_terms = terms(text)
            overlap = (
                len(query_terms & unit_terms) / max(1, len(query_terms))
                if query_terms
                else 0.5
            )
            score = 0.55 * overlap + 0.30 * base + _role_boost(role, field)
            if protected:
                score += 0.35
            ranked_units.append((score, text, field, protected))
        ranked_units.sort(key=lambda value: (value[3], value[0]), reverse=True)
        item_score = max((value[0] for value in ranked_units), default=base)
        item_score += 0.05 / (rank + 1)
        ranked_items.append((item_score, item, ranked_units))
    ranked_items.sort(key=lambda value: value[0], reverse=True)

    projected: list[dict[str, Any]] = []
    source_ids: list[str] = []
    used_tokens = 0
    for item_score, item, ranked_units in ranked_items:
        if len(projected) >= effective_max_items:
            break
        chosen: list[dict[str, Any]] = []
        chosen_text: set[str] = set()
        ordinary_count = 0
        for score, text, field, protected in ranked_units:
            if not protected and ordinary_count >= effective_sentence_limit:
                continue
            identity = " ".join(text.casefold().split())
            if identity in chosen_text:
                continue
            cost = estimate_tokens(text)
            if used_tokens + cost > budget_tokens and not protected:
                continue
            # Protected units still obey a hard 110% ceiling so one malformed
            # memory cannot crowd out the entire agent prompt.
            if used_tokens + cost > int(budget_tokens * 1.10):
                continue
            chosen_text.add(identity)
            chosen.append(
                {
                    "text": text,
                    "field": field,
                    "score": round(score, 6),
                    "protected": protected,
                }
            )
            used_tokens += cost
            if not protected:
                ordinary_count += 1
        if not chosen:
            continue
        memory_id = _memory_id(item)
        compact = {key: item[key] for key in _ALLOWED_KEYS if key in item}
        chosen_by_field: dict[str, list[str]] = {}
        for unit in chosen:
            chosen_by_field.setdefault(str(unit["field"]), []).append(
                str(unit["text"])
            )
        for field in (
            "constraints",
            "decisions",
            "user_corrections",
            "open_gaps",
            "successful_queries",
            "failed_queries",
        ):
            if field in compact:
                compact[field] = chosen_by_field.get(field, [])
        if "abstract" in compact:
            compact["abstract"] = " ".join(chosen_by_field.get("abstract", []))
        if "research_brief" in compact:
            compact["research_brief"] = " ".join(
                chosen_by_field.get("research_brief", [])
            )
        if "must_keep_spans" in compact:
            selected_exact = set(chosen_by_field.get("must_keep", []))
            compact["must_keep_spans"] = [
                span
                for span in compact["must_keep_spans"]
                if isinstance(span, dict)
                and str(span.get("text") or "") in selected_exact
            ]
        compact["memory_id"] = memory_id
        compact["summary"] = "\n".join(unit["text"] for unit in chosen)
        if str(compact.get("category") or "") in {
            "constraint",
            "correction",
            "project_decision",
        }:
            compact["user_corrections"] = [compact["summary"]]
        compact["selected_units"] = chosen
        compact["projection_score"] = round(item_score, 6)
        projected.append(compact)
        if memory_id:
            source_ids.append(memory_id)

    return ProjectionResult(
        items=projected,
        source_memory_ids=list(dict.fromkeys(source_ids)),
        raw_estimated_tokens=raw_tokens,
        estimated_tokens=used_tokens,
        level=level,
    )


def fit_text_values(
    values: list[str], *, query: str, limit: int, budget_tokens: int
) -> list[str]:
    """Keep relevant unique state instead of blindly slicing the newest N rows."""

    query_terms = terms(query)
    unique = list(dict.fromkeys(" ".join(value.split()).strip() for value in values if value.strip()))
    scored = []
    for recency, value in enumerate(reversed(unique)):
        overlap = len(query_terms & terms(value)) if query_terms else 1
        scored.append((overlap, -recency, value))
    scored.sort(reverse=True)
    result: list[str] = []
    used = 0
    for _, _, value in scored:
        cost = estimate_tokens(value)
        if result and used + cost > budget_tokens:
            continue
        result.append(value)
        used += cost
        if len(result) >= limit:
            break
    return result
