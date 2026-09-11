"""Durable conversation compaction and ephemeral memory projection."""

from .controller import CompactionController, CompactionPressure, estimate_tokens
from .models import CompactedSegment, MustKeepSpan, ProjectionResult, SegmentStateDelta
from .must_keep import extract_must_keep_spans, validate_must_keep_spans
from .projection import fit_text_values, project_memories
from .temporal import (
    authoritative_user_constraints,
    current_state_projection,
    separate_historical_segment_state,
)
from .segment import (
    COMPACTION_SYSTEM_PROMPT,
    apply_segment_to_rolling_state,
    build_compaction_payload,
    normalize_segment,
    render_rolling_summary,
    segment_document_content,
)

__all__ = [
    "CompactedSegment",
    "COMPACTION_SYSTEM_PROMPT",
    "CompactionController",
    "CompactionPressure",
    "MustKeepSpan",
    "ProjectionResult",
    "SegmentStateDelta",
    "apply_segment_to_rolling_state",
    "authoritative_user_constraints",
    "build_compaction_payload",
    "estimate_tokens",
    "extract_must_keep_spans",
    "fit_text_values",
    "current_state_projection",
    "normalize_segment",
    "project_memories",
    "render_rolling_summary",
    "separate_historical_segment_state",
    "segment_document_content",
    "validate_must_keep_spans",
]
