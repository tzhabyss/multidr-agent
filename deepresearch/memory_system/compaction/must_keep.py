"""Deterministic protection for exact conversation spans."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ...schemas import ConversationMessage
from ..extractor import MemoryExtractor
from .models import MustKeepKind, MustKeepSpan


_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.MULTILINE)
_CORRECTION = re.compile(
    r"(?:不对|不是|更正|纠正|改成|改为|修改为|放宽为|撤回|取消|作废|不再|而不是|actually|correction|"
    r"instead(?:\s+of)?|not\s+.+\s+but)",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(
    r"(?:必须|禁止|不要|不得|只能|仅使用|只使用|只允许|仅允许|以内|上限|至少|最多|优先|"
    r"只作为|仅作为|不代表|不能视为|不得视为|"
    r"不需要形成|无需形成|不构成|不需要记录|无需记录|"
    r"must|shall|should|do\s+not|don't|never|only|at\s+(?:least|most))",
    re.IGNORECASE,
)
_DECISION = re.compile(
    r"(?:决定|确定|采用|选用|方案是|统一使用|继续使用|新增|改成|设置为|固定为|"
    r"decid(?:e|ed)|use\s+.+(?:for|as)|switch(?:ed)?\s+to)",
    re.IGNORECASE,
)
_UNRESOLVED = re.compile(
    r"(?:尚未|还没|待确认|待解决|未解决|不确定|需要确认|"
    r"暂时不选|暂未选择|尚未选择|还没选择|待选择|"
    r"unresolved|unknown|to\s+be\s+confirmed|need(?:s)?\s+verification)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s)\]}>,;，；]+", re.IGNORECASE)
_NUMERIC = re.compile(
    r"(?:\b\d+(?:\.\d+)?%?\b|\b\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2})?日?\b|"
    r"\d+(?:\.\d+)?\s*(?:万|亿|k|m|gb|mb|token|tokens|元|美元|小时|天))",
    re.IGNORECASE,
)


def _kind_for(text: str) -> MustKeepKind | None:
    if _CORRECTION.search(text):
        return "correction"
    if _UNRESOLVED.search(text) or text.rstrip().endswith(("?", "？")):
        return "unresolved"
    if _CONSTRAINT.search(text):
        return "constraint"
    # Negated phrases such as "还没确定" contain the positive decision
    # marker "确定".  Unresolved must therefore win before decision.
    if _DECISION.search(text):
        return "decision"
    if _URL.search(text):
        return "citation"
    if _NUMERIC.search(text):
        return "numeric"
    return None


def extract_must_keep_spans(
    messages: Iterable[ConversationMessage], *, max_spans: int = 100
) -> tuple[list[MustKeepSpan], int]:
    """Return exact user-authored spans and a sensitive-span rejection count."""

    spans: list[MustKeepSpan] = []
    rejected_sensitive = 0
    seen: set[tuple[str, int, int, str]] = set()
    for message in messages:
        # Assistant/tool assertions are intentionally excluded to prevent them
        # from becoming protected user state through compaction.
        if message.role != "user":
            continue
        for match in _SENTENCE_PATTERN.finditer(message.content):
            raw = match.group(0)
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            if trailing <= leading:
                continue
            start = match.start() + leading
            end = match.start() + trailing
            text = message.content[start:end]
            kind = _kind_for(text)
            if kind is None:
                continue
            if MemoryExtractor._looks_secret(text):
                rejected_sensitive += 1
                continue
            identity = (message.message_id, start, end, kind)
            if identity in seen:
                continue
            seen.add(identity)
            spans.append(
                MustKeepSpan(
                    kind=kind,
                    text=text[:2000],
                    message_id=message.message_id,
                    start=start,
                    end=min(end, start + 2000),
                )
            )
            if len(spans) >= max_spans:
                return spans, rejected_sensitive
    return spans, rejected_sensitive


def validate_must_keep_spans(
    spans: Iterable[MustKeepSpan], messages: Iterable[ConversationMessage]
) -> list[MustKeepSpan]:
    """Fail closed on any span that is no longer an exact source substring."""

    by_id = {item.message_id: item for item in messages}
    validated: list[MustKeepSpan] = []
    for span in spans:
        source = by_id.get(span.message_id)
        if source is None or source.role != "user":
            continue
        if source.content[span.start : span.end] != span.text:
            continue
        validated.append(span)
    return validated
