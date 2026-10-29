"""Publication policy for original, successfully fetched research materials.

Collection names are derived from trusted request state, never model text. These
are application scopes in a trusted local store, not a replacement for API auth.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .store import digest


def research_collection(state) -> str | None:
    if not state:
        return None
    scope = [state.get("user_id"), state.get("project_id")]
    if any(not isinstance(value, str) or not value.strip() for value in scope):
        return None
    return "research:" + digest(json.dumps(scope, ensure_ascii=False))


def capture_metadata(document, state, published_at=None) -> dict | None:
    """Reject partial metadata, cached library documents and unsuccessful fetches."""
    uri = urlsplit(document.final_url)
    if (research_collection(state) is None or uri.scheme not in {"http", "https"}
            or not uri.hostname or uri.username or uri.password
            or not 200 <= document.http_status < 300 or not document.content.strip()
            or document.knowledge_ref
            or document.retrieval_method in {"academic_metadata", "academic_abstract"}):
        return None
    try:
        captured = datetime.fromisoformat(document.fetched_at.replace("Z", "+00:00"))
        if captured.tzinfo is None:
            return None
        captured_at = captured.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, AttributeError):
        return None
    return {
        "source_kind": "fetched_research_source",
        "first_fetched_at": captured_at,
        "last_fetched_at": captured_at,
        "origin_run_id": state.get("run_id"),
        "last_seen_run_id": state.get("run_id"),
        "requested_url": document.requested_url,
        "content_type": document.content_type,
        "published_at": published_at,
        "retrieval_method": document.retrieval_method,
    }


def capture_is_older(incoming: dict, current: dict) -> bool:
    # Both values were normalized to UTC by capture_metadata. Missing metadata
    # is possible for older/imported snapshots, which must not block first capture.
    a, b = incoming.get("last_fetched_at"), current.get("last_fetched_at")
    return bool(a and b and a < b)


def merge_capture_metadata(current: dict, incoming: dict) -> dict:
    merged = {**current, **incoming}
    for key in ("first_fetched_at", "origin_run_id"):
        if key in current:
            merged[key] = current[key]
    if not incoming.get("published_at") and current.get("published_at"):
        merged["published_at"] = current["published_at"]
    return merged
