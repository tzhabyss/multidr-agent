"""Replay oracle and review-gated Surrogate Verifier co-evolution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..ids import make_id, normalize_text
from ..llm import LLMJSONError, LLMResponse, ResearchLLM
from ..runtime_config import VerifierCoEvolutionRuntimeConfig
from .candidate import CandidateEvolutionService
from .models import CandidateRecord, EvolutionEpisode
from .store import CandidateStore, EvolutionEpisodeStore
from .verifier_models import (
    CandidateReplayObservation,
    CandidateReplayResult,
    OracleCheck,
    ProjectOracleResult,
    VerifierAssertion,
    VerifierGap,
    VerifierSuite,
    VerifierSuiteGenerationResult,
    VerifierSuiteRelease,
)
from .verifier_store import VerifierEvolutionStore


class VerifierCoEvolutionDisabled(RuntimeError):
    pass


class VerifierTokenBudgetExceeded(RuntimeError):
    pass


_BASELINE_HARD_CHECKS = (
    "schema_valid",
    "permissions_unchanged",
    "evidence_gates_preserved",
    "external_writes_blocked",
)

_TARGET_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "consecutive_no_evidence_gain": (
        ("evidence_gain", "increase"),
        ("zero_gain_searches", "decrease"),
    ),
    "duplicate_query": (
        ("duplicate_query_rate", "decrease"),
        ("duplicate_queries", "decrease"),
    ),
    "repeated_source_domain": (
        ("source_domain_diversity", "increase"),
        ("repeated_domain_rate", "decrease"),
    ),
    "source_family_repeated_failure": (
        ("source_family_failures", "decrease"),
        ("fetch_success_rate", "increase"),
    ),
    "fetch_route_failure": (
        ("fetch_route_failures", "decrease"),
        ("fetch_success_rate", "increase"),
    ),
    "extraction_without_binding": (
        ("evidence_bindings", "increase"),
        ("unbound_extractions", "decrease"),
    ),
    "skill_execution_failure": (("skill_execution_failures", "decrease"),),
    "weak_or_rejected_claim": (
        ("supported_claims", "increase"),
        ("weak_or_rejected_claims", "decrease"),
    ),
}

_UNIT_INTERVAL_METRICS = {
    "duplicate_query_rate",
    "repeated_domain_rate",
    "fetch_success_rate",
    "evidence_gain",
}

_ASSERTION_KINDS = {
    "required_hard_check",
    "metric_not_increase",
    "metric_not_decrease",
    "metric_strict_increase",
    "metric_strict_decrease",
}

_VERIFIER_GENERATOR_SYSTEM = """You improve a Deep Research Surrogate Verifier.
The Skill Candidate passed the existing surrogate checks but failed a separate
project oracle. Propose only additive, declarative assertions that would catch
the observable failure without revealing or encoding an answer. Never remove
or weaken prior assertions, execute code, add tools, change permissions, or
approve a Skill. Return one JSON object with an assertions array. Each item has
kind, subject, tolerance, severity, and rationale. kind must be one of:
required_hard_check, metric_not_increase, metric_not_decrease,
metric_strict_increase, metric_strict_decrease. subject must name one supplied
hard check or metric. Follow allowed_metric_assertions exactly: target metrics
must use their declared strict improvement direction, while token_cost and
model_calls may only use metric_not_increase. Never reward increased cost or
model calls. Keep the suite local to the supplied Skill and trigger."""


@dataclass
class _TokenMeter:
    limit: int
    input_tokens: int = 0
    output_tokens: int = 0

    def preflight(self, system: str, user: str, max_tokens: int) -> None:
        reservation = (
            len(system.encode("utf-8"))
            + len(user.encode("utf-8"))
            + int(max_tokens)
            + 4096
        )
        if self.input_tokens + self.output_tokens + reservation > self.limit:
            raise VerifierTokenBudgetExceeded(
                "Verifier evolution token cap would be exceeded"
            )

    def settle(self, response: LLMResponse) -> None:
        self.input_tokens += int(response.input_tokens)
        self.output_tokens += int(response.output_tokens)


def _version_patch(value: str) -> str:
    try:
        major, minor, patch = value.split(".", 2)
        return f"{int(major)}.{int(minor)}.{int(patch) + 1}"
    except (TypeError, ValueError):
        return "1.0.1"


def _baseline_suite(target_skill_id: str) -> VerifierSuite:
    return VerifierSuite(
        verifier_suite_id=make_id("verifier-suite", target_skill_id),
        version="1.0.0",
        target_skill_id=target_skill_id,
        assertions=[],
        status="builtin",
    )


def _metric_check(
    check_id: str,
    stable: float | None,
    candidate: float | None,
    *,
    relation: str,
    tolerance: float = 0.0,
) -> OracleCheck:
    if stable is None or candidate is None:
        return OracleCheck(
            check_id=check_id,
            status="unknown",
            summary="Replay did not provide both Stable and Candidate metrics.",
            stable_value=stable,
            candidate_value=candidate,
        )
    if relation == "increase":
        passed = candidate > stable + tolerance
    elif relation == "decrease":
        passed = candidate < stable - tolerance
    elif relation == "not_increase":
        passed = candidate <= stable + tolerance
    else:
        passed = candidate >= stable - tolerance
    return OracleCheck(
        check_id=check_id,
        status="pass" if passed else "fail",
        summary=f"Candidate must {relation.replace('_', ' ')} versus Stable.",
        stable_value=stable,
        candidate_value=candidate,
    )


def _target_metric_check(
    check_id: str,
    stable: float | None,
    candidate: float | None,
    *,
    metric_name: str,
    relation: str,
) -> OracleCheck:
    """Treat an already-optimal Stable target as no-headroom, not failure."""

    if stable is not None and candidate is not None:
        if relation == "decrease" and stable <= 0.0:
            return OracleCheck(
                check_id=check_id,
                status="unknown" if candidate <= stable else "fail",
                summary=(
                    "Stable is already at the metric floor; this Fixture cannot "
                    "demonstrate a strict Candidate improvement."
                ),
                stable_value=stable,
                candidate_value=candidate,
            )
        if (
            relation == "increase"
            and metric_name in _UNIT_INTERVAL_METRICS
            and stable >= 1.0
        ):
            return OracleCheck(
                check_id=check_id,
                status="unknown" if candidate >= stable else "fail",
                summary=(
                    "Stable is already at the metric ceiling; this Fixture cannot "
                    "demonstrate a strict Candidate improvement."
                ),
                stable_value=stable,
                candidate_value=candidate,
            )
    return _metric_check(
        check_id,
        stable,
        candidate,
        relation=relation,
    )


class ProjectOracleStack:
    """Evaluate only observable local properties; never claim report truth."""

    def __init__(self, *, max_cost_regression_ratio: float = 0.2) -> None:
        self.max_cost_regression_ratio = max_cost_regression_ratio

    def evaluate(
        self,
        observation: CandidateReplayObservation,
        *,
        episode: EvolutionEpisode,
        suite: VerifierSuite,
    ) -> ProjectOracleResult:
        checks: list[OracleCheck] = []
        checks.append(
            OracleCheck(
                check_id="external_writes_blocked",
                status="fail" if observation.external_writes_allowed else "pass",
                summary="Replay must not allow external writes.",
                candidate_value=observation.external_writes_allowed,
            )
        )
        for name in _BASELINE_HARD_CHECKS:
            if name == "external_writes_blocked":
                supplied = observation.hard_checks.get(name)
                if supplied is False:
                    checks.append(
                        OracleCheck(
                            check_id="reported_external_writes_blocked",
                            status="fail",
                            summary="Replay reported that external writes were not blocked.",
                            candidate_value=False,
                        )
                    )
                continue
            supplied = observation.hard_checks.get(name)
            checks.append(
                OracleCheck(
                    check_id=name,
                    status=("unknown" if supplied is None else ("pass" if supplied else "fail")),
                    summary=f"Required replay hard check: {name}.",
                    candidate_value=supplied,
                )
            )

        target_checks: list[OracleCheck] = []
        for trigger_type in episode.trigger_types:
            for metric_name, relation in _TARGET_METRICS.get(trigger_type, ()):
                if metric_name not in observation.stable_metrics and metric_name not in observation.candidate_metrics:
                    continue
                if observation.fixture_role == "guard":
                    guard_relation = (
                        "not_decrease" if relation == "increase" else "not_increase"
                    )
                    target_checks.append(
                        _metric_check(
                            f"guard:{metric_name}",
                            observation.stable_metrics.get(metric_name),
                            observation.candidate_metrics.get(metric_name),
                            relation=guard_relation,
                        )
                    )
                else:
                    target_checks.append(
                        _target_metric_check(
                            f"target:{metric_name}",
                            observation.stable_metrics.get(metric_name),
                            observation.candidate_metrics.get(metric_name),
                            metric_name=metric_name,
                            relation=relation,
                        )
                    )
        if target_checks:
            passed = any(item.status == "pass" for item in target_checks)
            regressed = any(item.status == "fail" for item in target_checks)
            unknown = any(item.status == "unknown" for item in target_checks)
            if observation.fixture_role == "guard":
                aggregate_status = (
                    "fail" if regressed else "unknown" if unknown else "pass"
                )
                aggregate_id = "guard_metrics_not_regressed"
                aggregate_summary = (
                    "Every observable trigger metric must be non-regressing on a Guard Fixture."
                )
            else:
                # Alternative target metrics are OR evidence for improvement,
                # but an observed regression cannot be hidden by another pass.
                aggregate_status = (
                    "fail" if regressed else "pass" if passed else "unknown"
                )
                aggregate_id = "target_problem_improved"
                aggregate_summary = (
                    "At least one trigger-specific local metric must strictly improve, "
                    "and no observable target metric may regress."
                    if passed or regressed
                    else "Stable has no measurable headroom on the supplied target metrics."
                )
            checks.append(
                OracleCheck(
                    check_id=aggregate_id,
                    status=aggregate_status,
                    summary=aggregate_summary,
                )
            )
            checks.extend(target_checks)
        else:
            checks.append(
                OracleCheck(
                    check_id="target_problem_improved",
                    status="unknown",
                    summary="Replay supplied no metric for the triggering local problem.",
                )
            )

        stable_cost = observation.stable_metrics.get("token_cost")
        candidate_cost = observation.candidate_metrics.get("token_cost")
        if stable_cost is not None and candidate_cost is not None:
            allowed = stable_cost * (1.0 + self.max_cost_regression_ratio)
            checks.append(
                OracleCheck(
                    check_id="cost_regression:token_cost",
                    status="pass" if candidate_cost <= allowed else "fail",
                    summary="Candidate token cost must remain within the configured regression ratio.",
                    stable_value=stable_cost,
                    candidate_value=candidate_cost,
                )
            )

        for assertion in suite.assertions:
            if assertion.applies_to_skill_ids and episode.target_skill_id not in assertion.applies_to_skill_ids:
                continue
            if assertion.trigger_types and not set(assertion.trigger_types).intersection(episode.trigger_types):
                continue
            if assertion.kind == "required_hard_check":
                supplied = observation.hard_checks.get(assertion.subject)
                checks.append(
                    OracleCheck(
                        check_id=f"suite:{assertion.assertion_id}",
                        status=("unknown" if supplied is None else ("pass" if supplied else "fail")),
                        summary=assertion.rationale,
                        candidate_value=supplied,
                    )
                )
                continue
            relation = {
                "metric_not_increase": "not_increase",
                "metric_not_decrease": "not_decrease",
                "metric_strict_increase": "increase",
                "metric_strict_decrease": "decrease",
            }[assertion.kind]
            checks.append(
                _metric_check(
                    f"suite:{assertion.assertion_id}",
                    observation.stable_metrics.get(assertion.subject),
                    observation.candidate_metrics.get(assertion.subject),
                    relation=relation,
                    tolerance=assertion.tolerance,
                )
            )

        decision_checks = [
            item
            for item in checks
            if not item.check_id.startswith(("target:", "guard:"))
        ]
        if any(item.status == "fail" for item in decision_checks):
            decision = "fail"
        elif any(item.status == "unknown" for item in decision_checks):
            decision = "hold"
        else:
            decision = "pass"
        return ProjectOracleResult(
            candidate_id=observation.candidate_id,
            replay_id=observation.replay_id,
            verifier_suite_id=suite.verifier_suite_id,
            verifier_suite_version=suite.version,
            decision=decision,
            checks=checks,
            summary={
                "pass": "Local replay passed the project oracle stack.",
                "fail": "Local replay exposed a surrogate-verifier gap or Candidate regression.",
                "hold": "Local replay lacks enough observable evidence for a decision.",
            }[decision],
        )


class VerifierCoEvolutionService:
    """Explicit CoEvo loop; never activates verifier or Skill versions by itself."""

    def __init__(
        self,
        *,
        candidate_service: CandidateEvolutionService,
        candidate_store: CandidateStore,
        episode_store: EvolutionEpisodeStore,
        verifier_store: VerifierEvolutionStore,
        generator_llm: ResearchLLM,
        config: VerifierCoEvolutionRuntimeConfig,
    ) -> None:
        self.candidate_service = candidate_service
        self.candidate_store = candidate_store
        self.episode_store = episode_store
        self.verifier_store = verifier_store
        self.generator_llm = generator_llm
        self.config = config
        self.oracle = ProjectOracleStack(
            max_cost_regression_ratio=config.max_cost_regression_ratio
        )

    def _suite_for(self, target_skill_id: str) -> VerifierSuite:
        return self.verifier_store.active_suite(target_skill_id) or _baseline_suite(
            target_skill_id
        )

    @staticmethod
    def _allowed_metric_assertions(gap: VerifierGap) -> dict[str, str]:
        observed: set[str] = set()
        for key in ("stable_metrics", "candidate_metrics"):
            metrics = gap.observable_failure.get(key, {})
            if isinstance(metrics, dict):
                observed.update(str(item) for item in metrics)
        allowed: dict[str, str] = {}
        relation_kind = {
            "increase": "metric_strict_increase",
            "decrease": "metric_strict_decrease",
        }
        for trigger_type in gap.trigger_types:
            for metric_name, relation in _TARGET_METRICS.get(trigger_type, ()):
                if metric_name in observed:
                    allowed[metric_name] = relation_kind[relation]
        for cost_metric in ("token_cost", "model_calls"):
            if cost_metric in observed:
                allowed[cost_metric] = "metric_not_increase"
        return allowed

    def evaluate_replay(
        self, observation: CandidateReplayObservation | dict[str, Any]
    ) -> CandidateReplayResult:
        if not self.config.enabled:
            raise VerifierCoEvolutionDisabled("Verifier co-evolution is disabled")
        replay = CandidateReplayObservation.model_validate(observation)
        existing = self.verifier_store.read_replay(replay.replay_id)
        if existing is not None:
            return existing.model_copy(update={"reused_existing": True})
        candidate = self.candidate_store.read(replay.candidate_id)
        if candidate.status != "accept_for_replay" or candidate.final_patch is None:
            raise ValueError("Only an accept_for_replay Candidate can be replayed")
        if candidate.source_episode_id != replay.source_episode_id:
            raise ValueError("Replay source Episode does not match Candidate")
        episode = self.episode_store.get_episode(replay.source_episode_id)
        if episode is None:
            raise LookupError(f"Evolution Episode not found: {replay.source_episode_id}")
        suite = self._suite_for(candidate.target_skill_id)
        oracle = self.oracle.evaluate(replay, episode=episode, suite=suite)
        gap_id = ""
        if oracle.decision == "fail":
            failed = [item for item in oracle.checks if item.status == "fail"]
            gap_id = make_id(
                "verifier-gap",
                candidate.candidate_id,
                replay.replay_id,
                suite.version,
                [item.check_id for item in failed],
            )
            gap = VerifierGap(
                gap_id=gap_id,
                candidate_id=candidate.candidate_id,
                replay_id=replay.replay_id,
                source_episode_id=episode.episode_id,
                target_skill_id=episode.target_skill_id,
                trigger_types=episode.trigger_types,
                verifier_suite_id=suite.verifier_suite_id,
                verifier_suite_version=suite.version,
                failed_check_ids=[item.check_id for item in failed],
                observable_failure={
                    "failed_checks": [item.model_dump(mode="json") for item in failed],
                    "stable_metrics": replay.stable_metrics,
                    "candidate_metrics": replay.candidate_metrics,
                    "hard_checks": replay.hard_checks,
                },
                evidence_refs=replay.evidence_refs,
                leak_safe_feedback="; ".join(item.summary for item in failed)[:2000],
            )
            self.verifier_store.write_gap(gap)
        result = CandidateReplayResult(
            replay_id=replay.replay_id,
            candidate_id=candidate.candidate_id,
            source_episode_id=episode.episode_id,
            status={"pass": "ready_for_shadow", "fail": "hold", "hold": "hold"}[
                oracle.decision
            ],
            observation=replay,
            oracle_result=oracle,
            verifier_gap_id=gap_id,
        )
        self.verifier_store.write_replay(result)
        return result

    async def generate_verifier_suite(
        self, gap_id: str
    ) -> VerifierSuiteGenerationResult:
        if not self.config.enabled:
            raise VerifierCoEvolutionDisabled("Verifier co-evolution is disabled")
        existing = self.verifier_store.find_suite_for_gap(gap_id)
        if existing is not None:
            return VerifierSuiteGenerationResult(
                suite=existing,
                reused_existing=True,
                store_path=str(
                    self.verifier_store.suite_path(
                        existing.verifier_suite_id, existing.version
                    )
                ),
            )
        gap = self.verifier_store.read_gap(gap_id)
        if gap is None:
            raise LookupError(f"Verifier Gap not found: {gap_id}")
        parent = self._suite_for(gap.target_skill_id)
        pending = self.verifier_store.pending_suite(
            gap.target_skill_id, parent.version
        )
        if pending is not None:
            raise ValueError(
                "Activate or discard the existing Verifier Suite Candidate "
                f"before generating another: {pending.verifier_suite_id}@{pending.version}"
            )
        payload = {
            "gap": gap.model_dump(mode="json"),
            "existing_assertions": [
                item.model_dump(mode="json") for item in parent.assertions
            ],
            "allowed_assertion_kinds": sorted(_ASSERTION_KINDS),
            "allowed_metric_assertions": self._allowed_metric_assertions(gap),
            "allowed_hard_checks": sorted(
                {
                    *_BASELINE_HARD_CHECKS,
                    *(
                        str(item)
                        for item in gap.observable_failure.get("hard_checks", {})
                    ),
                }
            ),
            "max_new_assertions": self.config.max_assertions_per_gap,
        }
        user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        meter = _TokenMeter(self.config.max_total_tokens_per_gap)
        meter.preflight(
            _VERIFIER_GENERATOR_SYSTEM, user, self.config.generator_max_tokens
        )
        try:
            value, response = await self.generator_llm.complete_json(
                _VERIFIER_GENERATOR_SYSTEM,
                user,
                max_tokens=self.config.generator_max_tokens,
            )
            meter.settle(response)
            assertions = self._parse_assertions(value, gap=gap, parent=parent)
        except (LLMJSONError, ValidationError, ValueError, VerifierTokenBudgetExceeded):
            raise
        suite = VerifierSuite(
            verifier_suite_id=parent.verifier_suite_id,
            version=_version_patch(parent.version),
            parent_version=parent.version,
            target_skill_id=gap.target_skill_id,
            assertions=[*parent.assertions, *assertions],
            source_gap_ids=list(dict.fromkeys([*parent.source_gap_ids, gap.gap_id])),
            status="candidate",
            generator_model_id=getattr(self.generator_llm, "model_id", ""),
            total_input_tokens=meter.input_tokens,
            total_output_tokens=meter.output_tokens,
        )
        path = self.verifier_store.write_suite(suite)
        return VerifierSuiteGenerationResult(suite=suite, store_path=str(path))

    def _parse_assertions(
        self,
        value: dict[str, Any],
        *,
        gap: VerifierGap,
        parent: VerifierSuite,
    ) -> list[VerifierAssertion]:
        raw = value.get("assertions")
        if not isinstance(raw, list) or not 1 <= len(raw) <= self.config.max_assertions_per_gap:
            raise ValueError(
                f"Verifier Generator must return 1..{self.config.max_assertions_per_gap} assertions"
            )
        metric_names = set()
        for key in ("stable_metrics", "candidate_metrics"):
            metrics = gap.observable_failure.get(key, {})
            if isinstance(metrics, dict):
                metric_names.update(str(item) for item in metrics)
        existing = {(item.kind, item.subject) for item in parent.assertions}
        allowed_metric_assertions = self._allowed_metric_assertions(gap)
        parsed: list[VerifierAssertion] = []
        for index, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                raise ValueError("Verifier assertion must be an object")
            kind = str(item.get("kind") or "")
            subject = normalize_text(str(item.get("subject") or ""))
            if kind not in _ASSERTION_KINDS or not subject:
                raise ValueError("Verifier assertion kind or subject is invalid")
            if kind == "required_hard_check":
                observed_hard_checks = gap.observable_failure.get("hard_checks", {})
                allowed_hard_checks = set(_BASELINE_HARD_CHECKS)
                if isinstance(observed_hard_checks, dict):
                    allowed_hard_checks.update(str(item) for item in observed_hard_checks)
                if subject not in allowed_hard_checks:
                    raise ValueError("Verifier assertion references an unobserved hard check")
            else:
                if subject not in metric_names:
                    raise ValueError("Verifier assertion references an unobserved metric")
                expected_kind = allowed_metric_assertions.get(subject)
                if expected_kind is None:
                    raise ValueError(
                        "Verifier assertion references a metric without a declared safe direction"
                    )
                if kind != expected_kind:
                    raise ValueError(
                        f"Unsafe metric direction for {subject}: {kind}; "
                        f"expected {expected_kind}"
                    )
            if (kind, subject) in existing or any(
                current.kind == kind and current.subject == subject for current in parsed
            ):
                raise ValueError("Verifier assertion duplicates an existing rule")
            tolerance = float(item.get("tolerance") or 0.0)
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("Verifier assertion tolerance must be finite and non-negative")
            rationale = normalize_text(str(item.get("rationale") or ""))
            if not rationale:
                raise ValueError("Verifier assertion requires a rationale")
            parsed.append(
                VerifierAssertion(
                    assertion_id=make_id(
                        "verifier-assertion", gap.gap_id, index, kind, subject
                    ),
                    kind=kind,
                    subject=subject,
                    applies_to_skill_ids=[gap.target_skill_id],
                    trigger_types=gap.trigger_types,
                    tolerance=tolerance,
                    severity=("warning" if item.get("severity") == "warning" else "error"),
                    rationale=rationale,
                    provenance_gap_ids=[gap.gap_id],
                )
            )
        return parsed

    def activate_verifier_suite(
        self,
        suite_id: str,
        version: str,
        *,
        human_review_approved: bool = False,
    ) -> VerifierSuiteRelease:
        if not self.config.enabled:
            raise VerifierCoEvolutionDisabled("Verifier co-evolution is disabled")
        if self.config.require_human_activation and not human_review_approved:
            raise PermissionError("Verifier Suite activation requires human approval")
        suite = self.verifier_store.read_suite(suite_id, version)
        if suite is None:
            raise LookupError(f"Verifier Suite not found: {suite_id}@{version}")
        active = self.verifier_store.active_release(suite.target_skill_id)
        expected_parent = active.version if active is not None else "1.0.0"
        if suite.parent_version != expected_parent:
            raise ValueError(
                f"Verifier Suite parent is stale: {suite.parent_version} != {expected_parent}"
            )
        release = VerifierSuiteRelease(
            release_id=make_id(
                "verifier-release", suite.verifier_suite_id, suite.version
            ),
            verifier_suite_id=suite.verifier_suite_id,
            version=suite.version,
            parent_version=suite.parent_version,
            target_skill_id=suite.target_skill_id,
            human_review_approved=human_review_approved,
            stable_skill_mutation_enabled=False,
        )
        self.verifier_store.write_release(release)
        return release

    async def revise_candidate_from_gap(self, gap_id: str):
        """Re-enter the Skill loop only after the evolved suite is active."""

        if not self.config.enabled:
            raise VerifierCoEvolutionDisabled("Verifier co-evolution is disabled")
        gap = self.verifier_store.read_gap(gap_id)
        if gap is None:
            raise LookupError(f"Verifier Gap not found: {gap_id}")
        suite = self._suite_for(gap.target_skill_id)
        if gap.gap_id not in suite.source_gap_ids:
            raise PermissionError(
                "Activate a Verifier Suite covering this Gap before revising the Skill"
            )
        assertions = [
            item
            for item in suite.assertions
            if gap.gap_id in item.provenance_gap_ids
        ]
        return await self.candidate_service.revise_for_verifier_gap(
            gap,
            assertions=[item.model_dump(mode="json") for item in assertions],
        )
