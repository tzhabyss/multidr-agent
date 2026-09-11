"""Scoped, expiring failure memory with corrective guidance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from .ids import make_id
from .schemas import FailureRecord
from .time_utils import now, now_iso


_FETCH_SUCCESS_STATUSES = {"fetched", "prefetched_unused"}
_FETCH_ROUTE_FAILURE_TYPES = {
    "url_unreachable",
    "transport_failure",
    "access_challenge",
    "dynamic_fetch_failure",
    "source_format_mismatch",
}
_FETCH_CONTENT_FAILURE_TYPES = {
    "target_content_missing",
    "extract_no_binding",
}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_active(record: FailureRecord, *, at: datetime | None = None) -> bool:
    if bool(record.metadata.get("resolved")):
        return False
    expiry = _parse_time(record.expires_at)
    if expiry is None:
        return True
    current = at or datetime.now(tz=expiry.tzinfo)
    if current.tzinfo is None and expiry.tzinfo is not None:
        current = current.replace(tzinfo=expiry.tzinfo)
    return current < expiry


def remember_failure(
    records: Iterable[FailureRecord | Mapping[str, Any]],
    *,
    scope_signature: str,
    failure_type: str,
    action_signature: str,
    scope_type: str = "task",
    scope_value: str = "",
    task_id: str = "",
    correction: str = "",
    corrective_route: str = "",
    source_family: str = "",
    expires_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[list[FailureRecord], FailureRecord]:
    """Upsert one failure by scoped action, retaining the useful correction."""

    parsed = [
        item if isinstance(item, FailureRecord) else FailureRecord.model_validate(item)
        for item in records
    ]
    identity = (scope_signature, failure_type, action_signature)
    current = next(
        (
            item
            for item in parsed
            if (item.scope_signature, item.failure_type, item.action_signature) == identity
        ),
        None,
    )
    timestamp = now_iso()
    incoming_metadata = dict(metadata or {})
    task_ids = {
        str(value)
        for value in incoming_metadata.get("task_ids", [])
        if str(value)
    }
    if task_id:
        task_ids.add(task_id)
    if task_ids:
        incoming_metadata["task_ids"] = sorted(task_ids)
    if current is None:
        updated = FailureRecord(
            failure_id=make_id("failure", *identity),
            scope_signature=scope_signature,
            scope_type=scope_type,
            scope_value=scope_value,
            task_id=task_id,
            failure_type=failure_type,
            action_signature=action_signature,
            correction=correction,
            corrective_route=corrective_route,
            source_family=source_family,
            expires_at=expires_at,
            metadata=incoming_metadata,
            first_seen_at=timestamp,
            last_seen_at=timestamp,
        )
        parsed.append(updated)
    else:
        merged_task_ids = {
            str(value)
            for value in current.metadata.get("task_ids", [])
            if str(value)
        }
        if current.task_id:
            merged_task_ids.add(current.task_id)
        merged_task_ids.update(task_ids)
        merged_metadata = {**current.metadata, **incoming_metadata}
        if merged_task_ids:
            merged_metadata["task_ids"] = sorted(merged_task_ids)
        updated = current.model_copy(
            update={
                "task_id": task_id or current.task_id,
                "correction": correction or current.correction,
                "corrective_route": corrective_route or current.corrective_route,
                "source_family": source_family or current.source_family,
                "recurrence_count": current.recurrence_count + 1,
                "postmortem_eligible": current.recurrence_count + 1 >= 2,
                "last_seen_at": timestamp,
                "expires_at": expires_at or current.expires_at,
                "metadata": merged_metadata,
            }
        )
        parsed = [updated if item.failure_id == current.failure_id else item for item in parsed]
    return parsed, updated


def active_failures(
    records: Iterable[FailureRecord | Mapping[str, Any]],
    *,
    scope_prefix: str = "",
    task_id: str = "",
) -> list[FailureRecord]:
    values: list[FailureRecord] = []
    for raw in records:
        item = raw if isinstance(raw, FailureRecord) else FailureRecord.model_validate(raw)
        if not is_active(item):
            continue
        if scope_prefix and not item.scope_signature.startswith(scope_prefix):
            continue
        if task_id:
            record_task_ids = {
                str(value)
                for value in item.metadata.get("task_ids", [])
                if str(value)
            }
            if item.task_id:
                record_task_ids.add(item.task_id)
            if record_task_ids and task_id not in record_task_ids:
                continue
        values.append(item)
    return values


def guidance_for_action(
    records: Iterable[FailureRecord | Mapping[str, Any]],
    action_signature: str,
    *,
    scope_prefix: str = "",
) -> list[str]:
    return list(
        dict.fromkeys(
            item.correction
            for item in active_failures(records, scope_prefix=scope_prefix)
            if item.action_signature == action_signature and item.correction
        )
    )


def resolve_failures_for_action(
    records: Iterable[FailureRecord | Mapping[str, Any]],
    *,
    action_signature: str,
    failure_types: set[str] | None = None,
    resolution: str = "subsequent_success",
) -> list[FailureRecord]:
    """Return same-ID updates that retire failures after a later success.

    Failure history remains auditable, but resolved records are excluded from
    active guidance and stopping sensors.
    """

    timestamp = now_iso()
    updates: list[FailureRecord] = []
    for raw in records:
        item = raw if isinstance(raw, FailureRecord) else FailureRecord.model_validate(raw)
        if item.action_signature != action_signature or not is_active(item):
            continue
        if failure_types is not None and item.failure_type not in failure_types:
            continue
        updates.append(
            item.model_copy(
                update={
                    "last_seen_at": timestamp,
                    "expires_at": timestamp,
                    "metadata": {
                        **item.metadata,
                        "resolved": True,
                        "resolved_at": timestamp,
                        "resolution": resolution,
                    },
                }
            )
        )
    return updates


def failures_from_fetch_traces(
    *,
    run_id: str,
    task_id: str,
    traces: Iterable[Mapping[str, Any]],
    existing: Iterable[FailureRecord | Mapping[str, Any]] = (),
    enhanced: bool = False,
) -> list[FailureRecord]:
    """Translate observable fetch failures without inferring semantic absence."""

    records: list[FailureRecord] = [
        item if isinstance(item, FailureRecord) else FailureRecord.model_validate(item)
        for item in existing
    ]
    updates: list[FailureRecord] = []
    for trace in traces:
        status = str(trace.get("status") or "")
        error = str(trace.get("error") or "")
        url = str(trace.get("url") or "")
        action_signature = f"fetch:{url}"
        content_chars = int(trace.get("content_chars") or 0)
        accepted_claims = int(trace.get("accepted_claims") or 0)
        if status in _FETCH_SUCCESS_STATUSES:
            # A successful prefetch is transport success even when the Worker
            # ultimately consumes another page. `prefetch_not_consumed` is a
            # lifecycle marker, not a network error.
            if status == "fetched" or content_chars > 0:
                updates.extend(
                    resolve_failures_for_action(
                        records,
                        action_signature=action_signature,
                        failure_types=_FETCH_ROUTE_FAILURE_TYPES,
                        resolution="fetch_succeeded",
                    )
                )
            if status == "fetched" and accepted_claims > 0:
                updates.extend(
                    resolve_failures_for_action(
                        records,
                        action_signature=action_signature,
                        failure_types=_FETCH_CONTENT_FAILURE_TYPES,
                        resolution="evidence_extracted",
                    )
                )
            continue
        # Only terminal fetch failures and pages that yielded no evidence are
        # failure-memory inputs. Other lifecycle statuses may carry diagnostic
        # strings but must not poison future routing.
        if status not in {"failed", "no_evidence"}:
            continue
        dynamic = bool(trace.get("dynamic_attempted"))
        lowered = error.casefold()
        access_error = any(
            marker in lowered
            for marker in ("403", "429", "forbidden", "challenge", "captcha", "blocked")
        )
        dynamic = dynamic or any(
            marker in lowered
            for marker in ("targetclosed", "dynamic", "browser", "page closed")
        )
        if not enhanced:
            failure_type = (
                "extract_no_binding"
                if status == "no_evidence"
                else "access_challenge"
                if access_error
                else "dynamic_fetch_failure"
                if dynamic and status == "failed"
                else "url_unreachable"
            )
            correction = (
                "Do not retry this exact route; change source family and try a reader, DOI/repository copy, PDF, mirror, or official alternate."
                if access_error or dynamic
                else "Do not blindly retry this exact route; inspect whether it is transient, then prefer a different source route if possible."
                if status == "failed"
                else "Keep the source as navigation and try a more field-specific source or extraction."
            )
            ttl_seconds = 900 if failure_type in {"dynamic_fetch_failure", "url_unreachable"} else 1800
            records, updated = remember_failure(
                records,
                scope_signature=f"run:{run_id}/task:{task_id}",
                failure_type=failure_type,
                action_signature=f"fetch:{url}",
                task_id=task_id,
                correction=correction,
                expires_at=(now() + timedelta(seconds=ttl_seconds)).isoformat(),
                metadata={
                    "url": url,
                    "error": error,
                    "error_class": error.split(":", 1)[0] if error else failure_type,
                    "provider": trace.get("provider"),
                    "dynamic_attempted": dynamic,
                    "requirement_ids": [
                        str(value)
                        for value in trace.get("requirement_ids", [])
                        if str(value)
                    ],
                    "route_action": "change_source_route" if access_error or dynamic else "inspect_then_retry_or_change_route",
                },
            )
            updates.append(updated)
            continue

        format_mismatch = any(
            marker in lowered
            for marker in ("unsupported mime", "source format", "pdf parse", "decode")
        )
        failure_type = (
            "source_format_mismatch"
            if format_mismatch
            else "target_content_missing"
            if status == "no_evidence" and content_chars > 0
            else "extract_no_binding"
            if status == "no_evidence"
            else "access_challenge"
            if access_error
            else "dynamic_fetch_failure"
            if dynamic and status == "failed"
            else "transport_failure"
        )
        correction = (
            "Do not retry this exact route; change source family and try a reader, DOI/repository copy, PDF, mirror, or official alternate."
            if access_error or dynamic
            else "Do not blindly retry this exact route; inspect whether it is transient, then prefer a different source route if possible."
            if status == "failed"
            else "Keep the source as navigation and try a more field-specific source or extraction."
        )
        ttl_seconds = 900 if failure_type in {"dynamic_fetch_failure", "transport_failure"} else 1800
        expiry = (now() + timedelta(seconds=ttl_seconds)).isoformat()
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.casefold()
        source_family = str(trace.get("provider") or domain or "unknown")
        route = (
            "alternate_access_route"
            if access_error or dynamic
            else "field_specific_source"
            if status == "no_evidence"
            else "alternate_source"
        )
        # Store one canonical event per failed URL/action. Task, domain, and
        # source-family scopes are compact indexes on the event rather than
        # four independent records that inflate state and stopping signals.
        scope_indexes = {
            "task": f"run:{run_id}/task:{task_id}",
            "url": f"url:{url}",
        }
        if domain:
            scope_indexes["domain"] = f"domain:{domain}"
        if source_family:
            scope_indexes["source_family"] = f"source-family:{source_family}"
        records, updated = remember_failure(
            records,
            scope_signature=f"url:{url}",
            scope_type="url",
            scope_value=url,
            failure_type=failure_type,
            action_signature=action_signature,
            task_id=task_id,
            correction=correction,
            corrective_route=route,
            source_family=source_family,
            expires_at=expiry,
            metadata={
                "url": url,
                "domain": domain,
                "scope_indexes": scope_indexes,
                "error": error,
                "error_class": error.split(":", 1)[0] if error else failure_type,
                "provider": trace.get("provider"),
                "dynamic_attempted": dynamic,
                "requirement_ids": [
                    str(value)
                    for value in trace.get("requirement_ids", [])
                    if str(value)
                ],
                "route_action": route,
                "semantic_absence_inferred": False,
                "resolved": False,
                "resolved_at": None,
            },
        )
        updates.append(updated)
    # A delta may observe the same action more than once. Keep only the newest
    # update per canonical record before it enters the global reducer.
    return list({item.failure_id: item for item in updates}.values())


def failures_from_search_traces(
    *,
    run_id: str,
    task_id: str,
    traces: Iterable[Mapping[str, Any]],
    existing: Iterable[FailureRecord | Mapping[str, Any]] = (),
) -> list[FailureRecord]:
    """Remember failed and zero-result query angles without declaring facts false."""

    records = [
        item if isinstance(item, FailureRecord) else FailureRecord.model_validate(item)
        for item in existing
    ]
    updates: list[FailureRecord] = []
    seen_queries: set[str] = set()
    for trace in traces:
        query = " ".join(str(trace.get("query") or "").casefold().split())
        if not query:
            continue
        equivalent = " ".join(sorted(query.split()))
        duplicate = equivalent in seen_queries
        seen_queries.add(equivalent)
        error = str(trace.get("error") or "")
        results = trace.get("results") or []
        if not duplicate and not error and results:
            continue
        failure_type = "duplicate_query" if duplicate else "skill_failure" if error else "query_no_gain"
        correction = (
            "Change language, source family, entity anchor, or relation angle before searching again."
        )
        records, updated = remember_failure(
            records,
            scope_signature=f"query-angle:{equivalent}",
            scope_type="query_angle",
            scope_value=equivalent,
            failure_type=failure_type,
            action_signature=f"search:{equivalent}",
            task_id=task_id,
            correction=correction,
            corrective_route="change_query_angle",
            expires_at=(now() + timedelta(seconds=1800)).isoformat(),
            metadata={
                "query": query,
                "error": error,
                "result_count": len(results),
                "semantic_absence_inferred": False,
                "run_id": run_id,
            },
        )
        updates.append(updated)
    return updates


def failures_from_extraction_traces(
    *,
    task_id: str,
    traces: Iterable[Mapping[str, Any]],
    existing: Iterable[FailureRecord | Mapping[str, Any]] = (),
) -> list[FailureRecord]:
    records = [
        item if isinstance(item, FailureRecord) else FailureRecord.model_validate(item)
        for item in existing
    ]
    updates: list[FailureRecord] = []
    for trace in traces:
        if int(trace.get("accepted_count") or 0) > 0:
            continue
        error = str(trace.get("error") or "")
        rejected = trace.get("rejected_candidates") or []
        if not error and not rejected:
            continue
        url = str(trace.get("source_uri") or "")
        records, updated = remember_failure(
            records,
            scope_signature=f"url:{url}",
            scope_type="url",
            scope_value=url,
            failure_type="extract_no_binding",
            action_signature=f"extract:{url}",
            task_id=task_id,
            correction="Try focused fields or a different page section, then change source if no target content is present.",
            corrective_route="focused_extract_then_alternate_source",
            expires_at=(now() + timedelta(seconds=1800)).isoformat(),
            metadata={
                "error": error,
                "rejected_count": len(rejected),
                "semantic_absence_inferred": False,
            },
        )
        updates.append(updated)
    return updates
