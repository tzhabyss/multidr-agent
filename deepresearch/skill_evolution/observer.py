"""Normalize existing runtime observations into SkillX-Lite Episodes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from ..failure_memory import active_failures
from ..ids import make_id
from ..skill_registry import SkillDefinition, SkillRegistry
from ..time_utils import now_iso
from .models import (
    EvolutionEpisode,
    EvolutionEvidenceReference,
    EvolutionSignal,
    EvolutionSignalType,
    candidate_policy_for,
)


_FETCH_ROUTE_FAILURES = {
    "url_unreachable",
    "transport_failure",
    "access_challenge",
    "dynamic_fetch_failure",
    "source_format_mismatch",
    "target_content_missing",
}


def _skill_identity(
    registry: SkillRegistry,
    skill_id: str,
) -> tuple[str, str, str, str]:
    definition = registry.get(skill_id) or registry.get(skill_id.replace("_", "-"))
    if definition is None:
        # Unknown executable tools are treated conservatively.  Observation is
        # still useful, but it must not become an automatic Candidate.
        return skill_id.replace("_", "-"), "unversioned", "access", "functional"
    return (
        definition.skill_id,
        definition.skill_version,
        definition.skill_layer,
        definition.capability_granularity,
    )


def _ref(path: str, record_type: str, record_id: Any = "") -> EvolutionEvidenceReference:
    return EvolutionEvidenceReference(
        artifact_path=path,
        record_type=record_type,
        record_id=str(record_id or ""),
    )


def _state_observed_at(state: Mapping[str, Any]) -> str:
    for event in reversed(list(state.get("events", []))):
        if event.get("timestamp"):
            return str(event["timestamp"])
    started_at = state.get("started_at")
    if isinstance(started_at, (int, float)):
        return datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    return now_iso()


class SkillEvolutionObserver:
    """Pure phase-one observer over a completed research state."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        duplicate_query_threshold: float = 0.7,
        repeated_domain_threshold: float = 0.65,
        max_signals_per_run: int = 100,
    ) -> None:
        self.registry = registry
        self.duplicate_query_threshold = duplicate_query_threshold
        self.repeated_domain_threshold = repeated_domain_threshold
        self.max_signals_per_run = max_signals_per_run

    def _signal(
        self,
        *,
        state: Mapping[str, Any],
        signal_type: EvolutionSignalType,
        source_key: str,
        target_skill_id: str,
        summary: str,
        confidence: float,
        severity: str = "warning",
        task_id: str = "",
        window_id: str = "",
        scope_type: str = "task",
        scope_value: str = "",
        context: Mapping[str, Any] | None = None,
        evidence_refs: Iterable[EvolutionEvidenceReference] = (),
    ) -> EvolutionSignal:
        skill_id, version, layer, granularity = _skill_identity(
            self.registry, target_skill_id
        )
        run_id = str(state.get("run_id") or "")
        return EvolutionSignal(
            signal_id=make_id(
                "skill-signal", run_id, signal_type, source_key, skill_id
            ),
            signal_type=signal_type,
            run_id=run_id,
            task_id=task_id,
            window_id=window_id,
            target_skill_id=skill_id,
            target_skill_version=version,
            runtime_layer=layer,
            capability_granularity=granularity,
            candidate_policy=candidate_policy_for(granularity),
            scope_type=scope_type,
            scope_value=scope_value,
            severity=severity,
            confidence=confidence,
            summary=summary,
            context=dict(context or {}),
            evidence_refs=list(evidence_refs),
            observed_at=_state_observed_at(state),
        )

    def collect_signals(self, state: Mapping[str, Any]) -> list[EvolutionSignal]:
        signals: list[EvolutionSignal] = []
        snapshots = list(state.get("sensor_history", []))
        for index, current in enumerate(snapshots):
            window_id = str(current.get("window_id") or current.get("snapshot_id") or index)
            snapshot_id = str(current.get("snapshot_id") or window_id)
            previous = snapshots[index - 1] if index else None
            if previous is not None:
                no_binding_gain = int(current.get("bound_evidence_count", 0)) <= int(
                    previous.get("bound_evidence_count", 0)
                )
                search_advanced = int(current.get("query_count", 0)) > int(
                    previous.get("query_count", 0)
                )
                if no_binding_gain and search_advanced:
                    signals.append(
                        self._signal(
                            state=state,
                            signal_type="consecutive_no_evidence_gain",
                            source_key=snapshot_id,
                            target_skill_id="web-search",
                            summary="Search activity increased without a new evidence binding.",
                            confidence=0.82,
                            window_id=window_id,
                            scope_type="run",
                            scope_value=str(state.get("run_id") or ""),
                            context={
                                "query_delta": int(current.get("query_count", 0))
                                - int(previous.get("query_count", 0)),
                                "binding_delta": int(current.get("bound_evidence_count", 0))
                                - int(previous.get("bound_evidence_count", 0)),
                            },
                            evidence_refs=[
                                _ref(
                                    "details/search/sensor-history.json",
                                    "sensor_snapshot",
                                    snapshot_id,
                                )
                            ],
                        )
                    )
            duplicate_ratio = float(current.get("duplicate_query_ratio", 0.0))
            if int(current.get("query_count", 0)) >= 2 and duplicate_ratio >= self.duplicate_query_threshold:
                signals.append(
                    self._signal(
                        state=state,
                        signal_type="duplicate_query",
                        source_key=snapshot_id,
                        target_skill_id="web-search",
                        summary="The local search window contains too many duplicate queries.",
                        confidence=min(0.98, 0.55 + duplicate_ratio / 2),
                        window_id=window_id,
                        scope_type="run",
                        scope_value=str(state.get("run_id") or ""),
                        context={"duplicate_query_ratio": duplicate_ratio},
                        evidence_refs=[
                            _ref("details/search/sensor-history.json", "sensor_snapshot", snapshot_id),
                            _ref("details/traces/search.json", "search_trace"),
                        ],
                    )
                )
            domain_ratio = float(current.get("repeated_domain_ratio", 0.0))
            if domain_ratio >= self.repeated_domain_threshold:
                signals.append(
                    self._signal(
                        state=state,
                        signal_type="repeated_source_domain",
                        source_key=snapshot_id,
                        target_skill_id="web-search",
                        summary="Fetched sources are concentrated in repeated domains.",
                        confidence=min(0.96, 0.55 + domain_ratio / 2),
                        window_id=window_id,
                        scope_type="run",
                        scope_value=str(state.get("run_id") or ""),
                        context={"repeated_domain_ratio": domain_ratio},
                        evidence_refs=[
                            _ref("details/search/sensor-history.json", "sensor_snapshot", snapshot_id),
                            _ref("details/traces/fetch.json", "fetch_trace"),
                        ],
                    )
                )
            repeated_families = int(
                current.get("repeated_source_family_failure_count", 0)
            )
            if repeated_families > 0:
                signals.append(
                    self._signal(
                        state=state,
                        signal_type="source_family_repeated_failure",
                        source_key=snapshot_id,
                        target_skill_id="safe-fetch",
                        summary="A source family failed through more than one access route.",
                        confidence=0.9,
                        window_id=window_id,
                        scope_type="source_family",
                        context={"repeated_source_family_count": repeated_families},
                        evidence_refs=[
                            _ref("details/search/sensor-history.json", "sensor_snapshot", snapshot_id),
                            _ref("details/search/failure-memory.json", "failure_record"),
                        ],
                    )
                )

        for failure in active_failures(state.get("failure_memory", [])):
            failure_type = failure.failure_type
            if failure_type in _FETCH_ROUTE_FAILURES:
                signal_type: EvolutionSignalType = "fetch_route_failure"
                target = "safe-fetch"
            elif failure_type == "extract_no_binding":
                signal_type = "extraction_without_binding"
                target = "evidence-extraction"
            elif failure_type == "skill_failure":
                signal_type = "skill_execution_failure"
                target = str(failure.metadata.get("skill_id") or failure.action_signature)
                target = target.removeprefix("skill:")
            elif failure_type == "rejected_claim":
                signal_type = "weak_or_rejected_claim"
                target = "evidence-extraction"
            elif failure_type in {"query_no_gain", "duplicate_query"}:
                signal_type = (
                    "consecutive_no_evidence_gain"
                    if failure_type == "query_no_gain"
                    else "duplicate_query"
                )
                target = "web-search"
            else:
                continue
            signals.append(
                self._signal(
                    state=state,
                    signal_type=signal_type,
                    source_key=failure.failure_id,
                    target_skill_id=target,
                    summary=f"Failure memory recorded {failure_type}.",
                    confidence=min(0.98, 0.76 + failure.recurrence_count * 0.06),
                    severity="error" if failure.recurrence_count >= 2 else "warning",
                    task_id=failure.task_id,
                    scope_type=(
                        failure.scope_type
                        if failure.scope_type in {"run", "task", "url", "source_family"}
                        else "task"
                    ),
                    scope_value=failure.scope_value,
                    context={
                        "failure_type": failure_type,
                        "action_signature": failure.action_signature,
                        "correction": failure.correction,
                        "corrective_route": failure.corrective_route,
                        "recurrence_count": failure.recurrence_count,
                    },
                    evidence_refs=[
                        _ref(
                            "details/search/failure-memory.json",
                            "failure_record",
                            failure.failure_id,
                        )
                    ],
                )
            )

        for trace in state.get("fetch_traces", []):
            trace_id = str(trace.get("trace_id") or make_id("fetch", trace))
            status = str(trace.get("status") or "")
            task_id = str(trace.get("task_id") or "")
            url = str(trace.get("url") or "")
            if status == "failed":
                signals.append(
                    self._signal(
                        state=state,
                        signal_type="fetch_route_failure",
                        source_key=trace_id,
                        target_skill_id="safe-fetch",
                        summary="A selected fetch route failed.",
                        confidence=0.94,
                        severity="error",
                        task_id=task_id,
                        scope_type="url",
                        scope_value=url,
                        context={"url": url, "error": str(trace.get("error") or "")[:600]},
                        evidence_refs=[_ref("details/traces/fetch.json", "fetch_trace", trace_id)],
                    )
                )
            if status in {"fetched", "no_evidence"} and int(
                trace.get("accepted_claims") or 0
            ) == 0 and int(trace.get("content_chars") or 0) > 0:
                signals.append(
                    self._signal(
                        state=state,
                        signal_type="extraction_without_binding",
                        source_key=trace_id,
                        target_skill_id="evidence-extraction",
                        summary="Fetched content produced no accepted claim or evidence binding.",
                        confidence=0.9,
                        task_id=task_id,
                        scope_type="url",
                        scope_value=url,
                        context={
                            "url": url,
                            "content_chars": int(trace.get("content_chars") or 0),
                            "accepted_claims": 0,
                        },
                        evidence_refs=[
                            _ref("details/traces/fetch.json", "fetch_trace", trace_id),
                            _ref("details/traces/extraction.json", "extraction_trace"),
                        ],
                    )
                )

        for trace in state.get("agent_loop_traces", []):
            trace_id = str(trace.get("trace_id") or make_id("loop", trace))
            for index, observation in enumerate(trace.get("observations", [])):
                if bool(observation.get("success", True)):
                    continue
                skill_name = str(observation.get("name") or "unknown-skill")
                signals.append(
                    self._signal(
                        state=state,
                        signal_type="skill_execution_failure",
                        source_key=f"{trace_id}:{index}",
                        target_skill_id=skill_name,
                        summary=f"Skill {skill_name} failed during the AgentLoop.",
                        confidence=0.99,
                        severity="error",
                        task_id=str(trace.get("task_id") or ""),
                        scope_type="task",
                        scope_value=str(trace.get("task_id") or ""),
                        context={
                            "iteration": int(trace.get("iteration") or 0),
                            "skill_name": skill_name,
                            "error": str(observation.get("error") or "")[:600],
                        },
                        evidence_refs=[
                            _ref("details/traces/agent-loop.json", "agent_loop_trace", trace_id)
                        ],
                    )
                )

        claim_signals: list[tuple[str, str, str, str]] = []
        claim_signals.extend(
            (
                str(item.get("link_id") or ""),
                "claim_link",
                str(item.get("claim_id") or ""),
                str(item.get("entailment_status") or ""),
            )
            for item in state.get("links", [])
            if item.get("entailment_status") in {"weak", "rejected"}
        )
        claim_signals.extend(
            (
                str(item.get("binding_id") or ""),
                "evidence_binding",
                str(item.get("requirement_id") or item.get("evidence_id") or ""),
                str(item.get("status") or ""),
            )
            for item in state.get("evidence_bindings", [])
            if item.get("status") in {"uncertain", "rejected"}
        )
        claim_signals.extend(
            (
                str(item.get("claim_id") or ""),
                "claim",
                str(item.get("claim_id") or ""),
                str(item.get("status") or ""),
            )
            for item in state.get("claims", [])
            if item.get("status") == "needs_verification"
        )
        for record_id, record_type, claim_id, status in claim_signals:
            path = (
                "details/evidence/claim-evidence-links.json"
                if record_type == "claim_link"
                else "details/search/citation-matrix.json"
                if record_type == "evidence_binding"
                else "details/evidence/claims.json"
            )
            signals.append(
                self._signal(
                    state=state,
                    signal_type="weak_or_rejected_claim",
                    source_key=f"{record_type}:{record_id}",
                    target_skill_id="evidence-extraction",
                    summary=f"A claim-level evidence decision was {status}.",
                    confidence=0.9 if status == "rejected" else 0.78,
                    severity="error" if status == "rejected" else "warning",
                    scope_type="claim",
                    scope_value=claim_id,
                    context={"record_type": record_type, "status": status},
                    evidence_refs=[_ref(path, record_type, record_id)],
                )
            )

        unique: dict[str, EvolutionSignal] = {}
        for signal in signals:
            unique.setdefault(signal.signal_id, signal)
        return list(unique.values())[: self.max_signals_per_run]

    def build_episodes(
        self, signals: Iterable[EvolutionSignal]
    ) -> list[EvolutionEpisode]:
        episodes: list[EvolutionEpisode] = []
        for signal in signals:
            action = {
                key: signal.context[key]
                for key in ("skill_name", "action_signature", "url", "iteration")
                if key in signal.context and signal.context[key] not in {"", None}
            }
            record_ids = [
                item.record_id for item in signal.evidence_refs if item.record_id
            ]
            episodes.append(
                EvolutionEpisode(
                    episode_id=make_id("evolution-episode", signal.signal_id),
                    run_id=signal.run_id,
                    task_id=signal.task_id,
                    target_skill_id=signal.target_skill_id,
                    target_skill_version=signal.target_skill_version,
                    runtime_layer=signal.runtime_layer,
                    capability_granularity=signal.capability_granularity,
                    candidate_policy=signal.candidate_policy,
                    trigger_signal_ids=[signal.signal_id],
                    trigger_types=[signal.signal_type],
                    scope={
                        "type": signal.scope_type,
                        "value": signal.scope_value,
                        "window_id": signal.window_id,
                    },
                    local_context={
                        "summary": signal.summary,
                        "signal_context": signal.context,
                    },
                    actions=[action] if action else [],
                    local_result={
                        "severity": signal.severity,
                        "observed_failure_or_low_gain": True,
                    },
                    evidence_refs=signal.evidence_refs,
                    replay_spec={
                        "artifact_paths": list(
                            dict.fromkeys(item.artifact_path for item in signal.evidence_refs)
                        ),
                        "record_ids": record_ids,
                        "external_writes_allowed": False,
                    },
                    confidence=signal.confidence,
                    created_at=signal.observed_at,
                )
            )
        return episodes

    def observe(
        self, state: Mapping[str, Any]
    ) -> tuple[list[EvolutionSignal], list[EvolutionEpisode]]:
        signals = self.collect_signals(state)
        return signals, self.build_episodes(signals)
