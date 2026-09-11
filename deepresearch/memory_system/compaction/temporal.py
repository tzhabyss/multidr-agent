"""Separate current state from immutable historical conversation evidence.

Conversation segments are append-only audit records.  Their state-bearing
fields therefore describe what was true *at that point in time*, not what is
necessarily true now.  This module keeps those records searchable while
preventing their old constraints and decisions from being promoted back into
the authoritative prompt state.
"""

from __future__ import annotations

from typing import Any

from ...schemas import SessionMemory


_STATE_FIELDS = ("constraints", "decisions", "user_corrections", "open_gaps")
_HISTORICAL_MARKER = "[HISTORICAL EVIDENCE - NOT CURRENT STATE]"


def current_state_projection(session: SessionMemory) -> dict[str, Any]:
    """Return the one authoritative session-state channel for an agent packet."""

    state = session.rolling_state
    return {
        "authority": "session.rolling_state",
        "session_id": session.session_id,
        "revision": state.revision,
        "source_segment_ids": list(state.segment_ids),
        "constraints": list(state.constraints),
        "decisions": list(state.decisions),
        # Corrections are an audit trail rather than mutable state.  They are
        # exposed separately so callers do not accidentally treat them as an
        # active constraint.
        "corrections": list(state.corrections),
        "unresolved": list(state.unresolved),
    }


def separate_historical_segment_state(
    items: list[dict[str, Any]], *, session_id: str
) -> list[dict[str, Any]]:
    """Mark immutable segments as history and remove their authority fields.

    The summary remains available as explicitly labelled evidence for queries
    such as "what did we use before?".  Current constraints, decisions and
    unresolved items are supplied independently by ``current_state_projection``.
    Non-segment memories (including active durable facts) are left unchanged.
    """

    separated: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        if item.get("memory_type") != "conversation_segment":
            separated.append(item)
            continue

        historical_fields = [field for field in _STATE_FIELDS if item.get(field)]
        same_session = str(item.get("session_id") or "") == session_id
        item.update(
            {
                "temporal_status": (
                    "current_session_history" if same_session else "cross_session_history"
                ),
                "state_authority": "historical_evidence_only",
                "state_projection_allowed": False,
                "historical_state_fields": historical_fields,
            }
        )
        for field in _STATE_FIELDS:
            if field in item:
                item[field] = []

        for field in ("abstract", "summary"):
            value = str(item.get(field) or "").strip()
            if value and not value.startswith(_HISTORICAL_MARKER):
                item[field] = f"{_HISTORICAL_MARKER} {value}"
        separated.append(item)
    return separated


def authoritative_user_constraints(
    session: SessionMemory, retrieved_items: list[dict[str, Any]]
) -> list[str]:
    """Merge active state and active facts without reviving Segment history."""

    retrieved = [
        value
        for item in retrieved_items
        if item.get("memory_type") != "conversation_segment"
        for value in [
            *item.get("constraints", []),
            *item.get("user_corrections", []),
        ]
        if str(value).strip()
    ]
    return list(
        dict.fromkeys(
            [
                *retrieved,
                *session.rolling_state.constraints,
                *session.user_corrections,
            ]
        )
    )[-30:]
