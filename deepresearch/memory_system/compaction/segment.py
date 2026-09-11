"""Normalize LLM output into an auditable, multi-resolution segment."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ...schemas import ConversationMessage, RollingMemoryState
from ..extractor import MemoryExtractor
from .controller import estimate_tokens
from .models import CompactedSegment, SegmentStateDelta
from .must_keep import extract_must_keep_spans, validate_must_keep_spans


COMPACTION_SYSTEM_PROMPT = (
    "You compress chat history into an auditable structured conversation "
    "segment for a research assistant. Return JSON only."
)


def build_compaction_payload(
    *,
    existing_state: RollingMemoryState,
    messages: list[ConversationMessage],
) -> dict[str, Any]:
    """Build the production compaction request shared with offline evals."""

    return {
        "existing_state": existing_state.model_dump(mode="json"),
        "messages": [
            {
                "message_id": item.message_id,
                "role": item.role,
                "content": item.content,
            }
            for item in messages
        ],
        "rules": [
            "Summarize facts about the conversation, not external facts as verified truth.",
            "Summarize only the supplied messages; do not rewrite the existing rolling state.",
            "Preserve user corrections, constraints, decisions, unresolved references, numbers, dates, citations, and active entities.",
            "Never promote assistant-authored claims into user facts.",
            "Keep only active constraints and decisions in add_* fields.",
            "Use retract_* when the user cancels a constraint or decision.",
            "Use supersede_* when a new constraint or decision replaces an old one.",
            "Never repeat credentials; retain only a credential non-storage policy.",
            "Return exact state changes separately from the narrative summary.",
        ],
        "output_schema": {
            "abstract": "50-100 token retrieval abstract",
            "segment_summary": "summary of only the supplied messages",
            "overview": {
                "active_entities": ["entity"],
                "constraints": ["durable constraint"],
                "decisions": ["accepted decision"],
                "corrections": ["user correction"],
                "unresolved": ["unresolved question or reference"],
            },
            "state_delta": {
                "add_constraints": ["constraint"],
                "add_decisions": ["decision"],
                "add_corrections": ["correction"],
                "add_unresolved": ["unresolved item"],
                "resolve_unresolved": ["resolved prior item"],
                "resolve_all_unresolved": "boolean; true only when every prior open item is resolved",
                "retract_constraints": ["cancelled prior constraint"],
                "retract_decisions": ["cancelled prior decision"],
                "supersede_constraints": [{"old": "old constraint", "new": "new constraint"}],
                "supersede_decisions": [{"old": "old decision", "new": "new decision"}],
            },
        },
    }


def _strings(value: Any, *, limit: int = 50) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        normalized = " ".join(str(item or "").split()).strip()
        if normalized and normalized not in result:
            result.append(normalized[:2000])
        if len(result) >= limit:
            break
    return result


def _first_sentence(value: str, *, max_chars: int = 400) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return ""
    for marker in ("。", "！", "？", ". ", "! ", "? "):
        position = normalized.find(marker)
        if position >= 0:
            return normalized[: min(position + len(marker.strip()), max_chars)].strip()
    return normalized[:max_chars].strip()


_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\b(?:sk-|ghp_|github_pat_|AKIA)[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"only_for_memorybench_not_a_real_credential", re.IGNORECASE),
    re.compile(
        r"((?:api[ _-]?key|access[ _-]?token|refresh[ _-]?token|password|"
        r"client[ _-]?secret|bearer|密码|口令|访问令牌)\s*(?:是|为|[:=]|值为)\s*)"
        r"[`'\"]?[^\s,，。；;]{4,}",
        re.IGNORECASE,
    ),
)
_SAFE_SECRET_POLICY = "用户要求不得记录或持久化敏感凭据；具体值已脱敏。"
_ONE_OFF_RESOLVED = re.compile(
    r"(?:一次性问题|临时问题|不需要形成(?:项目)?(?:约束|决定)|"
    r"无需形成(?:项目)?(?:约束|决定)|不需要记录|无需记录)",
    re.IGNORECASE,
)
_UNRESOLVED_TRANSITION = re.compile(
    r"(?:尚未|还没|待确认|待解决|未解决|不确定|需要确认|暂时不选|"
    r"暂未选择|尚未选择|还没选择|待选择|undecided|unresolved|to be confirmed)",
    re.IGNORECASE,
)


def _redact_sensitive_text(value: Any, *, narrative: bool = False) -> tuple[str, bool]:
    """Remove credential values without making the whole segment unusable."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return "", False
    redacted = text
    changed = False
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        updated, count = pattern.subn(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]",
            redacted,
        )
        redacted = updated
        changed = changed or count > 0
    # Labels plus a redacted assignment still intentionally look secret to the
    # generic detector.  Durable narrative stores the policy, never the label.
    if changed or MemoryExtractor._looks_secret(redacted):
        return (_SAFE_SECRET_POLICY if narrative else ""), True
    return redacted, False


def _safe_strings(value: Any, *, limit: int = 50) -> tuple[list[str], int]:
    result: list[str] = []
    redacted_count = 0
    for item in _strings(value, limit=limit):
        safe, redacted = _redact_sensitive_text(item)
        redacted_count += int(redacted)
        if safe and safe not in result:
            result.append(safe)
    return result, redacted_count


def _replacement_pairs(value: Any) -> tuple[list[dict[str, str]], int]:
    if not isinstance(value, list):
        return [], 0
    result: list[dict[str, str]] = []
    redacted_count = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        old, old_redacted = _redact_sensitive_text(item.get("old"))
        new, new_redacted = _redact_sensitive_text(item.get("new"))
        redacted_count += int(old_redacted) + int(new_redacted)
        if old and new:
            pair = {"old": old[:2000], "new": new[:2000]}
            if pair not in result:
                result.append(pair)
    return result, redacted_count


def _merge_delta_values(primary: Iterable[str], exact: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*primary, *exact]))[:100]


def _filter_delta_additions(
    values: Iterable[str],
    *,
    retractions: Iterable[str],
    replacements: Iterable[dict[str, str]],
) -> list[str]:
    """Drop stale add_* echoes without deleting a valid replacement.

    A replacement normally overlaps its predecessor (official-only ->
    official+peer-reviewed).  Therefore new-side matching must win before the
    broader old-side fuzzy check.
    """

    pairs = list(replacements)
    old_targets = [*retractions, *(item["old"] for item in pairs)]
    new_targets = [item["new"] for item in pairs]
    result: list[str] = []
    for item in values:
        if any(_matches_state_target(item, new) for new in new_targets):
            result.append(item)
            continue
        if any(_matches_state_target(item, old) for old in old_targets):
            continue
        result.append(item)
    return result


def normalize_segment(
    *,
    value: dict[str, Any],
    messages: list[ConversationMessage],
    segment_id: str,
    session_id: str,
    model_id: str,
    include_deterministic_spans: bool = True,
) -> CompactedSegment:
    """Accept both legacy and v2 LLM schemas and produce one strict segment."""

    overview = value.get("overview") if isinstance(value.get("overview"), dict) else {}
    summary_raw = str(
        value.get("segment_summary")
        or value.get("summary")
        or overview.get("summary")
        or ""
    ).strip()
    if not summary_raw:
        raise ValueError("compaction result omitted the segment summary")
    summary, summary_redacted = _redact_sensitive_text(summary_raw, narrative=True)
    abstract_raw = str(value.get("abstract") or overview.get("abstract") or "").strip()
    abstract, abstract_redacted = _redact_sensitive_text(abstract_raw, narrative=True)
    abstract = abstract or _first_sentence(summary)
    if not abstract:
        raise ValueError("compaction result omitted a usable abstract")

    source_spans, rejected_sensitive = extract_must_keep_spans(messages)
    source_spans = validate_must_keep_spans(source_spans, messages)
    spans = source_spans if include_deterministic_spans else []

    constraints, redacted_constraints = _safe_strings(
        value.get("constraints") or overview.get("constraints")
    )
    decisions, redacted_decisions = _safe_strings(
        value.get("decisions") or overview.get("decisions")
    )
    corrections, redacted_corrections = _safe_strings(
        value.get("user_corrections")
        or value.get("corrections")
        or overview.get("corrections")
    )
    unresolved, redacted_unresolved = _safe_strings(
        value.get("unresolved") or overview.get("unresolved")
    )
    active_entities, redacted_entities = _safe_strings(
        value.get("active_entities") or overview.get("active_entities"), limit=30
    )
    redacted_sensitive_fields = sum(
        (
            int(summary_redacted),
            int(abstract_redacted),
            redacted_constraints,
            redacted_decisions,
            redacted_corrections,
            redacted_unresolved,
            redacted_entities,
        )
    )
    if rejected_sensitive and _SAFE_SECRET_POLICY not in constraints:
        constraints.append(_SAFE_SECRET_POLICY)

    # Deterministic exact spans supplement LLM omissions. They do not promote
    # assistant-authored content or secrets into durable state.
    for span in spans:
        if span.kind == "constraint" and span.text not in constraints:
            constraints.append(span.text)
        elif span.kind == "decision" and span.text not in decisions:
            decisions.append(span.text)
        elif span.kind == "correction" and span.text not in corrections:
            corrections.append(span.text)
        elif span.kind == "unresolved" and span.text not in unresolved:
            unresolved.append(span.text)

    user_text = "\n".join(item.content for item in messages if item.role == "user")
    one_off_resolved = bool(_ONE_OFF_RESOLVED.search(user_text))
    # A correction can simultaneously open a new state slot ("cancel the DB
    # choice; leave it undecided"). Preserve that open item even though the
    # span's primary audit kind is correction.
    transition_unresolved = [
        span.text for span in spans if _UNRESOLVED_TRANSITION.search(span.text)
    ]
    if one_off_resolved:
        unresolved = []
        transition_unresolved = []
    else:
        unresolved = _merge_delta_values(unresolved, transition_unresolved)

    raw_delta = value.get("state_delta")
    raw_delta = raw_delta if isinstance(raw_delta, dict) else {}
    add_constraints, count = _safe_strings(raw_delta.get("add_constraints"))
    redacted_sensitive_fields += count
    add_decisions, count = _safe_strings(raw_delta.get("add_decisions"))
    redacted_sensitive_fields += count
    add_corrections, count = _safe_strings(raw_delta.get("add_corrections"))
    redacted_sensitive_fields += count
    add_unresolved, count = _safe_strings(raw_delta.get("add_unresolved"))
    redacted_sensitive_fields += count
    resolve_unresolved, count = _safe_strings(raw_delta.get("resolve_unresolved"))
    redacted_sensitive_fields += count
    retract_constraints, count = _safe_strings(raw_delta.get("retract_constraints"))
    redacted_sensitive_fields += count
    retract_decisions, count = _safe_strings(raw_delta.get("retract_decisions"))
    redacted_sensitive_fields += count
    supersede_constraints, count = _replacement_pairs(raw_delta.get("supersede_constraints"))
    redacted_sensitive_fields += count
    supersede_decisions, count = _replacement_pairs(raw_delta.get("supersede_decisions"))
    redacted_sensitive_fields += count

    exact_by_kind = {
        "constraint": [item.text for item in spans if item.kind == "constraint"],
        "decision": [item.text for item in spans if item.kind == "decision"],
        "correction": [item.text for item in spans if item.kind == "correction"],
        "unresolved": [
            item.text for item in spans if item.kind == "unresolved"
        ] + transition_unresolved,
    }
    effective_constraints = _merge_delta_values(
        add_constraints or constraints, exact_by_kind["constraint"]
    )
    if rejected_sensitive:
        effective_constraints = _merge_delta_values(
            effective_constraints, [_SAFE_SECRET_POLICY]
        )
    effective_decisions = _merge_delta_values(
        add_decisions or decisions, exact_by_kind["decision"]
    )
    effective_constraints = _filter_delta_additions(
        effective_constraints,
        retractions=retract_constraints,
        replacements=supersede_constraints,
    )
    effective_decisions = _filter_delta_additions(
        effective_decisions,
        retractions=retract_decisions,
        replacements=supersede_decisions,
    )

    delta = SegmentStateDelta(
        add_constraints=effective_constraints,
        add_decisions=effective_decisions,
        add_corrections=_merge_delta_values(
            add_corrections or corrections, exact_by_kind["correction"]
        ),
        add_unresolved=(
            []
            if one_off_resolved
            else _merge_delta_values(
                add_unresolved or unresolved, exact_by_kind["unresolved"]
            )
        ),
        resolve_unresolved=resolve_unresolved,
        resolve_all_unresolved=bool(raw_delta.get("resolve_all_unresolved", False)),
        retract_constraints=retract_constraints,
        retract_decisions=retract_decisions,
        supersede_constraints=supersede_constraints,
        supersede_decisions=supersede_decisions,
    )
    return CompactedSegment(
        segment_id=segment_id,
        abstract=abstract[:1200],
        summary=summary[:12000],
        active_entities=active_entities,
        constraints=constraints,
        decisions=decisions,
        corrections=corrections,
        unresolved=unresolved,
        must_keep_spans=spans,
        rejected_sensitive_spans=rejected_sensitive,
        redacted_sensitive_fields=redacted_sensitive_fields,
        state_delta=delta,
        detail_ref={
            "session_id": session_id,
            "message_ids": [item.message_id for item in messages],
        },
        compression={
            "model": model_id,
            "source_message_count": len(messages),
            "source_estimated_tokens": estimate_tokens(
                "\n".join(item.content for item in messages)
            ),
        },
    )


def _merge(existing: Iterable[str], additions: Iterable[str], *, limit: int) -> list[str]:
    return list(dict.fromkeys([*existing, *additions]))[-limit:]


def _state_key(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _matches_state_target(value: str, target: str) -> bool:
    left = _state_key(value)
    right = _state_key(target)
    if not left or not right:
        return False
    if left == right or (min(len(left), len(right)) >= 4 and (left in right or right in left)):
        return True
    # Bounded fuzzy fallback for a model-produced retraction that names the
    # same state item with a shorter phrase (for example "Embedding model
    # selection" versus a longer open question).  It is deliberately token
    # based and requires two shared units to avoid broad substring deletion.
    def tokens(text: str) -> set[str]:
        latin = set(re.findall(r"[a-z0-9@._-]{2,}", text.casefold()))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        grams = {chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))}
        return latin | grams

    def concepts(text: str) -> set[str]:
        lowered = text.casefold()
        mapping = {
            "only": ("only", "只使用", "只能", "仅使用", "只允许"),
            "official": ("official", "官方来源", "官方资料"),
            "peer-reviewed": ("peer-reviewed", "peer reviewed", "同行评审"),
            "blog": ("blog", "博客"),
            "personal": ("personal", "个人博客"),
            "prohibited": ("prohibited", "must not", "禁止", "不得"),
            "delivery": ("delivery", "交付"),
            "date": ("date", "日期"),
            "retrieval": ("retrieval", "召回"),
            "hybrid": ("hybrid", "混合检索", "融合"),
            "embedding": ("embedding", "向量模型"),
            "undecided": ("undecided", "待确认", "还没确定", "暂时不选", "待选择"),
        }
        result = {name for name, aliases in mapping.items() if any(alias in lowered for alias in aliases)}
        result.update(re.findall(r"\b(?:bm25|dense|graph|rrf|bge-m3|postgresql)\b", lowered))
        result.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", lowered))
        return result

    left_tokens = tokens(value)
    right_tokens = tokens(target)
    shared = left_tokens & right_tokens
    if len(shared) >= 2 and len(shared) / max(1, min(len(left_tokens), len(right_tokens))) >= 0.5:
        return True
    left_concepts = concepts(value)
    right_concepts = concepts(target)
    shared_concepts = left_concepts & right_concepts
    return len(shared_concepts) >= 2 and len(shared_concepts) / max(
        1, min(len(left_concepts), len(right_concepts))
    ) >= 0.6


def _remove_state_values(values: Iterable[str], targets: Iterable[str]) -> list[str]:
    target_list = list(targets)
    return [
        item
        for item in values
        if not any(_matches_state_target(item, target) for target in target_list)
    ]


def apply_segment_to_rolling_state(
    state: RollingMemoryState,
    segment: CompactedSegment,
    *,
    segment_limit: int,
) -> RollingMemoryState:
    constraints = _remove_state_values(
        state.constraints, segment.state_delta.retract_constraints
    )
    for replacement in segment.state_delta.supersede_constraints:
        constraints = _remove_state_values(constraints, [replacement["old"]])
        constraints = _merge(constraints, [replacement["new"]], limit=100)
    constraints = _merge(
        constraints, segment.state_delta.add_constraints, limit=100
    )

    decisions = _remove_state_values(
        state.decisions, segment.state_delta.retract_decisions
    )
    for replacement in segment.state_delta.supersede_decisions:
        old = replacement["old"]
        new = replacement["new"]
        decisions = _remove_state_values(decisions, [old])
        decisions = _merge(decisions, [new], limit=100)
    decisions = _merge(
        decisions, segment.state_delta.add_decisions, limit=100
    )
    unresolved = (
        []
        if segment.state_delta.resolve_all_unresolved
        else _remove_state_values(
            state.unresolved, segment.state_delta.resolve_unresolved
        )
    )
    unresolved = _merge(
        unresolved, segment.state_delta.add_unresolved, limit=100
    )
    return state.model_copy(
        update={
            "revision": state.revision + 1,
            "segment_ids": _merge(
                state.segment_ids, [segment.segment_id], limit=segment_limit
            ),
            "segment_abstracts": _merge(
                state.segment_abstracts, [segment.abstract], limit=segment_limit
            ),
            "constraints": constraints,
            "decisions": decisions,
            "corrections": _merge(
                state.corrections,
                segment.state_delta.add_corrections,
                limit=100,
            ),
            "unresolved": unresolved,
        }
    )


def render_rolling_summary(state: RollingMemoryState) -> str:
    """Build a bounded prompt projection without re-summarizing old prose."""

    sections: list[str] = []
    if state.segment_abstracts:
        sections.append("Recent history: " + " | ".join(state.segment_abstracts))
    if state.constraints:
        sections.append("Constraints: " + "; ".join(state.constraints))
    if state.decisions:
        sections.append("Decisions: " + "; ".join(state.decisions))
    if state.corrections:
        sections.append("Corrections: " + "; ".join(state.corrections))
    if state.unresolved:
        sections.append("Unresolved: " + "; ".join(state.unresolved))
    return "\n".join(sections)


def segment_document_content(segment: CompactedSegment) -> str:
    exact = [item.text for item in segment.must_keep_spans]
    sections = [
        segment.abstract,
        segment.summary,
        "Constraints: " + "; ".join(segment.constraints),
        "Decisions: " + "; ".join(segment.decisions),
        "Corrections: " + "; ".join(segment.corrections),
        "Unresolved: " + "; ".join(segment.unresolved),
        "Exact spans: " + "; ".join(exact),
    ]
    return "\n".join(
        item for item in sections if item.split(":", 1)[-1].strip()
    )
