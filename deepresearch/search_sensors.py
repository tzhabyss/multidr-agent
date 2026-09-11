"""Progress sensors for fail-soft strategy changes and stopping advice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse
import re

from .ids import make_id
from .failure_memory import active_failures


def _normalized_query(value: str) -> str:
    tokens = re.findall(r"[\w\u3400-\u9fff]+", value.casefold(), flags=re.UNICODE)
    return " ".join(sorted(dict.fromkeys(tokens)))


def progress_snapshot(
    state: Mapping[str, Any], *, window_id: str, enhanced: bool = False
) -> dict[str, Any]:
    successful_fetches = [
        item for item in state.get("fetch_traces", []) if item.get("status") == "fetched"
    ]
    failed_fetches = [
        item
        for item in state.get("fetch_traces", [])
        if item.get("status") in {"failed", "no_evidence"}
    ]
    queries = [
        str(item.get("query") or "").casefold()
        for item in state.get("search_traces", [])
        if item.get("query")
    ]
    unique_queries = set(_normalized_query(item) if enhanced else item for item in queries)
    snapshot = {
        "snapshot_id": make_id("sensor", state.get("run_id"), window_id),
        "window_id": window_id,
        "entity_count": len(state.get("entity_candidates", [])),
        "materialized_entity_count": sum(
            item.get("materialization_status") == "materialized"
            for item in state.get("entity_candidates", [])
        ),
        "bound_evidence_count": len(state.get("evidence_bindings", [])),
        "filled_cell_count": int(state.get("cell_coverage", {}).get("filled_cell_count", 0)),
        "conflict_count": sum(
            item.get("status") == "conflicting"
            for item in state.get("evidence_bindings", [])
        ),
        "independent_source_count": len(
            {
                item.get("provenance", {}).get("independent_domain")
                for item in state.get("sources", [])
                if item.get("provenance", {}).get("independent_domain")
            }
        ),
        "query_count": len(queries),
        "duplicate_query_ratio": (
            1.0 - len(unique_queries) / len(queries) if queries else 0.0
        ),
        "successful_fetch_count": len(successful_fetches),
        "failed_fetch_count": len(failed_fetches),
        "failed_fetch_ratio": (
            len(failed_fetches) / (len(successful_fetches) + len(failed_fetches))
            if successful_fetches or failed_fetches
            else 0.0
        ),
        "token_count": sum(
            int(item.get("input_tokens", 0)) + int(item.get("output_tokens", 0))
            for item in state.get("usage_events", [])
            if item.get("category") == "llm"
        ),
    }
    if not enhanced:
        return snapshot

    search_traces = list(state.get("search_traces", []))
    fetch_traces = list(state.get("fetch_traces", []))
    fetched_query_ids = {
        str(item.get("query_id") or "")
        for item in fetch_traces
        if item.get("query_id")
    }
    successful_query_ids = {
        str(item.get("query_id") or "")
        for item in fetch_traces
        if item.get("status") == "fetched" and item.get("query_id")
    }
    domains = [
        urlparse(str(item.get("url") or "")).netloc.casefold()
        for item in fetch_traces
        if item.get("url")
    ]
    domains = [item for item in domains if item]
    source_family_failures: dict[str, set[str]] = {}
    for record in active_failures(state.get("failure_memory", [])):
        item = record.model_dump(mode="json")
        family = str(item.get("source_family") or "")
        action = str(item.get("action_signature") or item.get("failure_id") or "")
        if family and action:
            source_family_failures.setdefault(family, set()).add(action)
    fetched_with_no_evidence = sum(
        item.get("status") in {"fetched", "no_evidence"}
        and int(item.get("accepted_claims") or 0) == 0
        for item in fetch_traces
    )
    snapshot.update(
        {
            "search_without_open_count": sum(
                bool(item.get("results"))
                and str(item.get("query_id") or "") not in fetched_query_ids
                for item in search_traces
            ),
            "opened_query_count": len(successful_query_ids),
            "open_without_evidence_count": fetched_with_no_evidence,
            "repeated_domain_ratio": (
                1.0 - len(set(domains)) / len(domains) if domains else 0.0
            ),
            "unique_url_count": len(
                {str(item.get("url") or "") for item in fetch_traces if item.get("url")}
            ),
            "repeated_source_family_failure_count": sum(
                len(actions) >= 2 for actions in source_family_failures.values()
            ),
        }
    )
    return snapshot


def _delta(current: Mapping[str, Any], previous: Mapping[str, Any], key: str) -> int:
    return max(0, int(current.get(key, 0)) - int(previous.get(key, 0)))


def evaluate_progress(
    history: Sequence[Mapping[str, Any]],
    *,
    minimum_windows: int = 2,
    minimum_stop_signals: int = 3,
    enhanced: bool = False,
) -> dict[str, Any]:
    """Advise an LLM controller; only multiple signals may recommend stopping."""

    if len(history) < 2:
        return {
            "status": "warming_up",
            "action": "continue",
            "signals": {},
            "reason": "At least two snapshots are needed to measure progress.",
        }
    current, previous = history[-1], history[-2]
    deltas = {
        "new_entities": _delta(current, previous, "entity_count"),
        "new_materialized_entities": _delta(
            current, previous, "materialized_entity_count"
        ),
        "new_bindings": _delta(current, previous, "bound_evidence_count"),
        "new_filled_cells": _delta(current, previous, "filled_cell_count"),
        "new_independent_sources": _delta(
            current, previous, "independent_source_count"
        ),
    }
    signals = {
        "no_semantic_gain": not any(deltas.values()),
        "high_duplicate_query_ratio": float(
            current.get("duplicate_query_ratio", 0.0)
        )
        >= 0.7,
        "high_failed_fetch_ratio": float(current.get("failed_fetch_ratio", 0.0))
        >= 0.7,
        "minimum_windows_reached": len(history) >= minimum_windows,
    }
    if enhanced:
        signals.update(
            {
                "search_without_open": int(current.get("search_without_open_count", 0))
                > 0,
                "open_without_new_evidence": (
                    int(current.get("open_without_evidence_count", 0))
                    > 0
                    and deltas["new_bindings"] == 0
                ),
                "repeated_domains": float(current.get("repeated_domain_ratio", 0.0))
                >= 0.65,
                "repeated_source_family_failures": int(
                    current.get("repeated_source_family_failure_count", 0)
                )
                > 0,
            }
        )
    value_signal_count = sum(
        value for key, value in signals.items() if key != "minimum_windows_reached"
    )
    if enhanced:
        recent = list(history[-3:])
        repeated_no_gain = len(recent) >= 3 and all(
            int(item.get("bound_evidence_count", 0))
            == int(recent[0].get("bound_evidence_count", 0))
            and int(item.get("filled_cell_count", 0))
            == int(recent[0].get("filled_cell_count", 0))
            for item in recent[1:]
        )
        independent_low_value_signals = sum(
            bool(value)
            for key, value in signals.items()
            if key
            not in {
                "minimum_windows_reached",
                "no_semantic_gain",
            }
        )
        if signals["no_semantic_gain"] and not repeated_no_gain:
            action = "change_strategy"
            status = "stalled"
            reason = (
                "Low gain detected once; change source family, language, entity anchor, "
                "or relation angle before considering drain."
            )
        elif (
            signals["minimum_windows_reached"]
            and repeated_no_gain
            and independent_low_value_signals >= minimum_stop_signals
        ):
            action = "consider_drain"
            status = "low_value"
            reason = (
                "Repeated no-gain windows and independent loop signals agree; "
                "the controller may drain this branch."
            )
        else:
            action = "continue"
            status = "progressing" if any(deltas.values()) else "watch"
            reason = "Continue; stopping evidence is not yet independently repeated."
    elif signals["no_semantic_gain"] and len(history) == minimum_windows:
        action = "change_strategy"
        status = "stalled"
        reason = "No new typed artifacts appeared; change query or source family first."
    elif (
        signals["minimum_windows_reached"]
        and value_signal_count >= minimum_stop_signals
    ):
        action = "consider_drain"
        status = "low_value"
        reason = "Several independent low-value signals agree; the controller may drain this branch."
    else:
        action = "continue"
        status = "progressing" if any(deltas.values()) else "watch"
        reason = "Progress exists or the evidence for stopping is not yet strong enough."
    return {
        "status": status,
        "action": action,
        "deltas": deltas,
        "signals": signals,
        "value_signal_count": value_signal_count,
        "reason": reason,
    }
