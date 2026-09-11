"""Bounded Generator-Verifier workflow for CoEvoSkills-Lite phase two."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..ids import make_id, normalize_text, stable_hash
from ..llm import LLMJSONError, LLMResponse, ResearchLLM
from ..runtime_config import CandidateGenerationRuntimeConfig
from ..skill_registry import SkillRegistry
from .capsule import CandidateCapsuleBuilder, CandidateCapsuleError
from .models import (
    CandidateGenerationResult,
    CandidatePatch,
    CandidateRecord,
    CandidateRevision,
    CandidateVerification,
    EvolutionEpisode,
    PatchOperation,
    VerifierIssue,
)
from .store import CandidateStore, EvolutionEpisodeStore
from .verifier_models import VerifierGap


class CandidateGenerationDisabled(RuntimeError):
    pass


class CandidateTokenBudgetExceeded(RuntimeError):
    pass


_GENERATOR_SYSTEM = """You are the Candidate Generator in CoEvoSkills-Lite.
Create one small patch to the target Skill from the supplied local capsule.
You may only propose add, delete, or replace operations. Never rewrite the
whole Skill, add tools or permissions, bypass evidence/citation gates, or edit
another Skill. Preserve the parent Skill's safety constraints. Return exactly
one JSON object with: target_skill_id, parent_skill_version,
capability_granularity, trigger_conditions, applicability_preconditions,
operations, expected_improvements, risks, source_episode_ids,
requested_tools, requested_permissions, gate_bypasses. Each operation contains
kind, target_section, anchor, old_text, new_text, and rationale. For add, anchor
must be an exact substring of the parent excerpt. For delete/replace, old_text
must be an exact substring. Keep the combined changed text short."""


_REVISION_SYSTEM = """You are the Candidate Generator revising a bounded Skill
patch. Use only the original local capsule, the previous patch, and the fixed
Verifier issues. Make the smallest change that addresses those issues. Keep the
same target Skill and parent version. Return the same JSON structure as the
original generation request. Every operation in the revised patch is applied
directly to the unchanged parent Skill, not on top of the previous patch. Never
use text introduced only by the previous patch as old_text or an anchor. To
revise an add operation, normally keep its exact parent anchor and replace the
operation's new_text with the complete corrected addition. Copy every replace
or delete old_text exactly from the parent excerpt. Do not add permissions,
tools, or gate bypasses."""


_VERIFIER_SYSTEM = """You are the independent, fixed CoEvoSkills-Lite Semantic
Verifier. You did not generate this patch. Judge only local properties; do not
claim that the complete research answer is true. Check over-generalization,
missing preconditions, conflict with the parent Skill, permission or tool
expansion, evidence/citation bypass, forbidden strategic changes, and whether
the patch can be replayed from the supplied Episode. A Skill patch is allowed
to add a new, locally scoped capability or term that is absent from a minimal
parent Skill; absence alone is not a parent conflict. Treat an explicit,
episode-supported evidence checklist as compatible when its preconditions are
narrow and it does not weaken the parent's literal-scope or evidence-floor
requirements. Require a real contradiction before setting parent_compatible
false. Return exactly one JSON
object with decision (accept_for_replay, revise, hold, or reject), summary,
issues, and checks. Every issue contains code, severity, message, suggested_fix.
checks must contain booleans for: overgeneralization_safe,
preconditions_complete, parent_compatible, permissions_unchanged,
evidence_gates_preserved, replayable. accept_for_replay is allowed only when all
six checks are true and there are no error issues."""


@dataclass
class _TokenMeter:
    limit: int
    input_tokens: int = 0
    output_tokens: int = 0

    def preflight(self, system: str, user: str, max_tokens: int) -> None:
        # UTF-8 bytes plus framing and the complete output allowance is a
        # deliberately conservative reservation for one sequential call.
        reservation = (
            len(system.encode("utf-8"))
            + len(user.encode("utf-8"))
            + int(max_tokens)
            + 4096
        )
        if self.input_tokens + self.output_tokens + reservation > self.limit:
            raise CandidateTokenBudgetExceeded(
                "Candidate token cap would be exceeded: "
                f"used={self.input_tokens + self.output_tokens}, "
                f"next_reservation={reservation}, limit={self.limit}"
            )

    def settle(self, response: LLMResponse) -> None:
        self.input_tokens += int(response.input_tokens)
        self.output_tokens += int(response.output_tokens)


def _fingerprint(patch: CandidatePatch) -> str:
    operations = [
        {
            "kind": item.kind,
            "target_section": normalize_text(item.target_section).casefold(),
            "anchor": normalize_text(item.anchor).casefold(),
            "old_text": normalize_text(item.old_text).casefold(),
            "new_text": normalize_text(item.new_text).casefold(),
        }
        for item in patch.operations
    ]
    return stable_hash(
        patch.target_skill_id,
        patch.parent_skill_version,
        patch.capability_granularity,
        operations,
        [normalize_text(item).casefold() for item in patch.applicability_preconditions],
    )


def _string_list(value: Any) -> list[str]:
    """Accept a single model string without accidentally splitting characters."""

    if isinstance(value, str):
        return [value] if normalize_text(value) else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if normalize_text(str(item))]


def _patch_from_payload(
    value: dict[str, Any],
    *,
    candidate_id: str,
    episode: EvolutionEpisode,
) -> CandidatePatch:
    raw_operations = value.get("operations")
    if not isinstance(raw_operations, list):
        raw_operations = []
    operations = [
        PatchOperation(
            operation_id=make_id("patch-op", candidate_id, index, raw),
            kind=str(raw.get("kind") or ""),
            target_section=str(raw.get("target_section") or ""),
            anchor=str(raw.get("anchor") or ""),
            old_text=str(raw.get("old_text") or ""),
            new_text=str(raw.get("new_text") or ""),
            rationale=str(raw.get("rationale") or ""),
        )
        for index, raw in enumerate(raw_operations, 1)
        if isinstance(raw, dict)
    ]
    patch = CandidatePatch(
        candidate_id=candidate_id,
        # Identity fields are code-owned. Missing model copies default to the
        # Episode, while explicit conflicting values remain visible to the
        # deterministic mismatch checks below.
        target_skill_id=str(
            value.get("target_skill_id") or episode.target_skill_id
        ),
        parent_skill_version=str(
            value.get("parent_skill_version") or episode.target_skill_version
        ),
        capability_granularity=str(
            value.get("capability_granularity")
            or episode.capability_granularity
        ),
        trigger_conditions=_string_list(value.get("trigger_conditions")),
        applicability_preconditions=_string_list(
            value.get("applicability_preconditions")
        ),
        operations=operations,
        expected_improvements=_string_list(value.get("expected_improvements")),
        risks=_string_list(value.get("risks")),
        source_episode_ids=(
            _string_list(value.get("source_episode_ids"))
            or [episode.episode_id]
        ),
        requested_tools=_string_list(value.get("requested_tools")),
        requested_permissions=_string_list(value.get("requested_permissions")),
        gate_bypasses=_string_list(value.get("gate_bypasses")),
    )
    return patch.model_copy(update={"patch_fingerprint": _fingerprint(patch)})


def _issue(code: str, message: str, *, severity: str = "error") -> VerifierIssue:
    return VerifierIssue(code=code, severity=severity, message=message)


def deterministic_guard(
    patch: CandidatePatch,
    *,
    episode: EvolutionEpisode,
    parent_text: str,
    max_operations: int,
    max_patch_chars: int,
) -> list[VerifierIssue]:
    """Reject structural/safety violations before semantic verification."""

    issues: list[VerifierIssue] = []
    if patch.target_skill_id != episode.target_skill_id:
        issues.append(_issue("target_skill_mismatch", "Patch targets another Skill."))
    if patch.parent_skill_version != episode.target_skill_version:
        issues.append(_issue("parent_version_mismatch", "Parent version changed."))
    if patch.capability_granularity != episode.capability_granularity:
        issues.append(_issue("granularity_mismatch", "Capability granularity changed."))
    if episode.capability_granularity == "strategic":
        issues.append(_issue("strategic_change_forbidden", "Strategic Skill changes are forbidden."))
    if episode.episode_id not in patch.source_episode_ids:
        issues.append(_issue("missing_source_episode", "Source Episode is not declared."))
    if not patch.trigger_conditions:
        issues.append(_issue("missing_trigger", "Patch has no trigger condition."))
    if not patch.applicability_preconditions:
        issues.append(_issue("missing_precondition", "Patch has no applicability precondition."))
    if not patch.expected_improvements:
        issues.append(_issue("missing_expected_improvement", "Patch has no local expected improvement."))
    if not 1 <= len(patch.operations) <= max_operations:
        issues.append(
            _issue(
                "operation_count",
                f"Patch must contain 1..{max_operations} operations.",
            )
        )
    if patch.requested_tools:
        issues.append(_issue("new_tools_forbidden", "Candidate requested additional tools."))
    if patch.requested_permissions:
        issues.append(_issue("permission_expansion", "Candidate requested permission changes."))
    if patch.gate_bypasses:
        issues.append(_issue("gate_bypass", "Candidate requested a gate bypass."))
    changed_chars = 0
    for operation in patch.operations:
        changed_chars += len(operation.old_text) + len(operation.new_text)
        if not normalize_text(operation.target_section):
            issues.append(_issue("missing_target_section", "Patch operation has no target section."))
        if not normalize_text(operation.rationale):
            issues.append(_issue("missing_rationale", "Patch operation has no rationale."))
        if operation.kind == "add":
            if not operation.new_text.strip():
                issues.append(_issue("empty_add", "Add operation has no new text."))
            if not operation.anchor or operation.anchor not in parent_text:
                issues.append(_issue("anchor_not_found", "Add anchor is not in the parent Skill."))
        elif operation.kind == "delete":
            if not operation.old_text or operation.old_text not in parent_text:
                issues.append(_issue("old_text_not_found", "Delete text is not in the parent Skill."))
            if operation.new_text.strip():
                issues.append(_issue("delete_has_new_text", "Delete operation contains new text."))
        elif operation.kind == "replace":
            if not operation.old_text or operation.old_text not in parent_text:
                issues.append(_issue("old_text_not_found", "Replace text is not in the parent Skill."))
            if not operation.new_text.strip():
                issues.append(_issue("empty_replace", "Replace operation has no new text."))
    if changed_chars > max_patch_chars:
        issues.append(
            _issue(
                "patch_too_large",
                f"Changed text uses {changed_chars} characters; limit is {max_patch_chars}.",
            )
        )
    return issues


def _verification_from_payload(
    value: dict[str, Any], response: LLMResponse
) -> CandidateVerification:
    raw_issues = value.get("issues")
    if not isinstance(raw_issues, list):
        raw_issues = []
    issues = [
        VerifierIssue(
            code=str(item.get("code") or "semantic_issue"),
            severity=("warning" if item.get("severity") == "warning" else "error"),
            message=str(item.get("message") or "Verifier reported an issue."),
            suggested_fix=str(item.get("suggested_fix") or ""),
        )
        for item in raw_issues
        if isinstance(item, dict)
    ]
    required_checks = (
        "overgeneralization_safe",
        "preconditions_complete",
        "parent_compatible",
        "permissions_unchanged",
        "evidence_gates_preserved",
        "replayable",
    )
    raw_checks = value.get("checks") if isinstance(value.get("checks"), dict) else {}
    checks = {key: bool(raw_checks.get(key, False)) for key in required_checks}
    decision = str(value.get("decision") or "hold")
    if decision not in {"accept_for_replay", "revise", "hold", "reject"}:
        decision = "hold"
    if decision == "accept_for_replay" and (
        not all(checks.values()) or any(item.severity == "error" for item in issues)
    ):
        decision = "revise"
        issues.append(
            _issue(
                "invalid_acceptance",
                "Verifier acceptance contradicted failed checks or error issues.",
            )
        )
    return CandidateVerification(
        decision=decision,
        summary=str(value.get("summary") or ""),
        issues=issues,
        checks=checks,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


class CandidateEvolutionService:
    """Explicit offline phase-two service; never mutates or loads Stable Skills."""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        episode_store: EvolutionEpisodeStore,
        candidate_store: CandidateStore,
        artifact_root: str | Path,
        generator_llm: ResearchLLM,
        verifier_llm: ResearchLLM,
        config: CandidateGenerationRuntimeConfig,
    ) -> None:
        self.registry = registry
        self.episode_store = episode_store
        self.candidate_store = candidate_store
        self.generator_llm = generator_llm
        self.verifier_llm = verifier_llm
        self.config = config
        self.capsules = CandidateCapsuleBuilder(
            registry=registry,
            episode_store=episode_store,
            artifact_root=artifact_root,
            max_similar_episodes=config.max_similar_episodes,
            max_capsule_chars=config.max_capsule_chars,
            parent_skill_excerpt_chars=config.parent_skill_excerpt_chars,
            artifact_excerpt_chars=config.artifact_excerpt_chars,
            max_patch_operations=config.max_patch_operations,
            max_patch_chars=config.max_patch_chars,
        )

    async def _call_json(
        self,
        llm: ResearchLLM,
        system: str,
        payload: dict[str, Any],
        *,
        max_tokens: int,
        meter: _TokenMeter,
    ) -> tuple[dict[str, Any], LLMResponse]:
        user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        meter.preflight(system, user, max_tokens)
        try:
            value, response = await llm.complete_json(
                system, user, max_tokens=max_tokens
            )
        except LLMJSONError as exc:
            meter.settle(exc.response)
            raise
        meter.settle(response)
        return value, response

    def _terminal_record(
        self,
        *,
        episode: EvolutionEpisode,
        candidate_id: str,
        status: str,
        reasons: list[str],
        capsule=None,
        revisions=None,
        final_patch=None,
        meter: _TokenMeter | None = None,
        parent_candidate_id: str = "",
        source_verifier_gap_ids: list[str] | None = None,
    ) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=candidate_id,
            parent_candidate_id=parent_candidate_id,
            source_episode_id=episode.episode_id,
            source_verifier_gap_ids=source_verifier_gap_ids or [],
            run_id=episode.run_id,
            target_skill_id=episode.target_skill_id,
            parent_skill_version=episode.target_skill_version,
            capability_granularity=episode.capability_granularity,
            status=status,
            capsule=capsule,
            revisions=revisions or [],
            final_patch=final_patch,
            final_reasons=reasons,
            generator_model_id=getattr(self.generator_llm, "model_id", ""),
            verifier_model_id=getattr(self.verifier_llm, "model_id", ""),
            total_input_tokens=meter.input_tokens if meter else 0,
            total_output_tokens=meter.output_tokens if meter else 0,
            stable_skill_mutation_enabled=False,
            production_loading_enabled=False,
        )

    def _persist(self, record: CandidateRecord) -> CandidateGenerationResult:
        path = self.candidate_store.write(record)
        return CandidateGenerationResult(record=record, store_path=str(path))

    async def generate_for_episode(
        self,
        episode_id: str,
        *,
        human_review_approved: bool = False,
    ) -> CandidateGenerationResult:
        if not self.config.enabled:
            raise CandidateGenerationDisabled(
                "CoEvoSkills-Lite candidate generation is disabled"
            )
        existing = self.candidate_store.find_for_episode(episode_id)
        if existing is not None:
            return CandidateGenerationResult(
                record=existing,
                reused_existing=True,
                store_path=str(
                    self.candidate_store.path_for_candidate(existing.candidate_id)
                ),
            )
        episode = self.episode_store.get_episode(episode_id)
        if episode is None:
            raise LookupError(f"Evolution Episode not found: {episode_id}")
        candidate_id = make_id(
            "candidate", episode.episode_id, episode.target_skill_id, episode.target_skill_version
        )
        if episode.capability_granularity == "strategic" or episode.candidate_policy == "observe_only":
            return self._persist(
                self._terminal_record(
                    episode=episode,
                    candidate_id=candidate_id,
                    status="reject",
                    reasons=["Strategic/observe-only Episode cannot generate a Candidate."],
                )
            )
        if (
            episode.candidate_policy == "episode_requires_review"
            and not human_review_approved
        ):
            return self._persist(
                self._terminal_record(
                    episode=episode,
                    candidate_id=candidate_id,
                    status="hold",
                    reasons=["Functional Episode requires explicit human review approval."],
                )
            )
        try:
            capsule = self.capsules.build(episode)
        except CandidateCapsuleError as exc:
            return self._persist(
                self._terminal_record(
                    episode=episode,
                    candidate_id=candidate_id,
                    status="hold",
                    reasons=[str(exc)],
                )
            )
        definition = self.registry.get(episode.target_skill_id)
        assert definition is not None
        parent_text = definition.load_instructions()
        meter = _TokenMeter(self.config.max_total_tokens_per_candidate)
        revisions: list[CandidateRevision] = []
        previous_patch: CandidatePatch | None = None
        final_patch: CandidatePatch | None = None
        final_status = "hold"
        final_reasons: list[str] = []
        verifier_issues: list[dict[str, Any]] = []
        try:
            for round_no in range(1, self.config.max_generator_verifier_rounds + 1):
                if round_no == 1:
                    generator_payload = {"capsule": capsule.model_dump(mode="json")}
                    generator_system = _GENERATOR_SYSTEM
                else:
                    generator_payload = {
                        "capsule": capsule.model_dump(mode="json"),
                        "previous_patch": previous_patch.model_dump(mode="json")
                        if previous_patch
                        else {},
                        "verifier_issues": verifier_issues,
                    }
                    generator_system = _REVISION_SYSTEM
                raw_patch, generator_response = await self._call_json(
                    self.generator_llm,
                    generator_system,
                    generator_payload,
                    max_tokens=self.config.generator_max_tokens,
                    meter=meter,
                )
                patch = _patch_from_payload(
                    raw_patch,
                    candidate_id=candidate_id,
                    episode=episode,
                )
                final_patch = patch
                changed = (
                    previous_patch is None
                    or patch.patch_fingerprint != previous_patch.patch_fingerprint
                )
                issues = deterministic_guard(
                    patch,
                    episode=episode,
                    parent_text=parent_text,
                    max_operations=self.config.max_patch_operations,
                    max_patch_chars=self.config.max_patch_chars,
                )
                if previous_patch is not None and not changed:
                    issues.append(
                        _issue(
                            "revision_no_change",
                            "Generator returned the same patch after a revise decision.",
                        )
                    )
                duplicate = self.candidate_store.query(
                    patch_fingerprint=patch.patch_fingerprint,
                    limit=1,
                )
                if duplicate and duplicate[0].source_episode_id != episode.episode_id:
                    issues.append(
                        _issue(
                            "duplicate_patch",
                            "An equivalent Candidate patch already exists.",
                        )
                    )
                if issues:
                    revisions.append(
                        CandidateRevision(
                            round_no=round_no,
                            patch=patch,
                            deterministic_issues=issues,
                            changed_from_previous=changed,
                            generator_input_tokens=generator_response.input_tokens,
                            generator_output_tokens=generator_response.output_tokens,
                        )
                    )
                    final_status = "reject"
                    final_reasons = [item.message for item in issues]
                    break
                verifier_payload = {
                    "capsule": capsule.model_dump(mode="json"),
                    "candidate_patch": patch.model_dump(mode="json"),
                }
                raw_verification, verifier_response = await self._call_json(
                    self.verifier_llm,
                    _VERIFIER_SYSTEM,
                    verifier_payload,
                    max_tokens=self.config.verifier_max_tokens,
                    meter=meter,
                )
                verification = _verification_from_payload(
                    raw_verification, verifier_response
                )
                revisions.append(
                    CandidateRevision(
                        round_no=round_no,
                        patch=patch,
                        verification=verification,
                        changed_from_previous=changed,
                        generator_input_tokens=generator_response.input_tokens,
                        generator_output_tokens=generator_response.output_tokens,
                    )
                )
                if verification.decision == "accept_for_replay":
                    final_status = "accept_for_replay"
                    final_reasons = [verification.summary]
                    break
                if verification.decision in {"hold", "reject"}:
                    final_status = verification.decision
                    final_reasons = [verification.summary, *[item.message for item in verification.issues]]
                    break
                previous_patch = patch
                verifier_issues = [
                    item.model_dump(mode="json") for item in verification.issues
                ]
                if round_no == self.config.max_generator_verifier_rounds:
                    final_status = "hold"
                    final_reasons = [
                        "Verifier still requested revision after the maximum two rounds."
                    ]
        except (CandidateTokenBudgetExceeded, LLMJSONError, ValidationError, ValueError) as exc:
            final_status = "hold"
            final_reasons = [f"{type(exc).__name__}: {exc}"]
        record = self._terminal_record(
            episode=episode,
            candidate_id=candidate_id,
            status=final_status,
            reasons=final_reasons,
            capsule=capsule,
            revisions=revisions,
            final_patch=final_patch,
            meter=meter,
        )
        return self._persist(record)

    async def revise_for_verifier_gap(
        self,
        gap: VerifierGap,
        *,
        assertions: list[dict[str, Any]],
    ) -> CandidateGenerationResult:
        """Create one immutable child Candidate from an activated verifier gap."""

        if not self.config.enabled:
            raise CandidateGenerationDisabled(
                "CoEvoSkills-Lite candidate generation is disabled"
            )
        parent = self.candidate_store.read(gap.candidate_id)
        if parent.status != "accept_for_replay" or parent.final_patch is None:
            raise ValueError("Verifier-gap revision requires an accepted parent Candidate")
        if parent.capsule is None:
            raise ValueError("Parent Candidate has no replayable input capsule")
        episode = self.episode_store.get_episode(gap.source_episode_id)
        if episode is None:
            raise LookupError(f"Evolution Episode not found: {gap.source_episode_id}")
        candidate_id = make_id(
            "candidate-verifier-revision",
            parent.candidate_id,
            gap.gap_id,
            assertions,
        )
        existing_path = self.candidate_store.path_for_candidate(candidate_id)
        if existing_path.is_file():
            return CandidateGenerationResult(
                record=self.candidate_store.read(candidate_id),
                reused_existing=True,
                store_path=str(existing_path),
            )
        meter = _TokenMeter(self.config.max_total_tokens_per_candidate)
        revisions: list[CandidateRevision] = []
        final_patch: CandidatePatch | None = None
        final_status = "hold"
        final_reasons: list[str] = []
        try:
            raw_patch, generator_response = await self._call_json(
                self.generator_llm,
                _REVISION_SYSTEM,
                {
                    "capsule": parent.capsule.model_dump(mode="json"),
                    "previous_patch": parent.final_patch.model_dump(mode="json"),
                    "verifier_gap": {
                        "gap_id": gap.gap_id,
                        "failed_check_ids": gap.failed_check_ids,
                        "feedback": gap.leak_safe_feedback,
                    },
                    "verifier_assertions": assertions,
                },
                max_tokens=self.config.generator_max_tokens,
                meter=meter,
            )
            patch = _patch_from_payload(
                raw_patch,
                candidate_id=candidate_id,
                episode=episode,
            )
            final_patch = patch
            issues = deterministic_guard(
                patch,
                episode=episode,
                parent_text=self.registry.get(episode.target_skill_id).load_instructions(),
                max_operations=self.config.max_patch_operations,
                max_patch_chars=self.config.max_patch_chars,
            )
            changed = patch.patch_fingerprint != parent.final_patch.patch_fingerprint
            if not changed:
                issues.append(
                    _issue(
                        "verifier_gap_revision_no_change",
                        "Generator did not change the patch after an evolved verifier failure.",
                    )
                )
            duplicate = self.candidate_store.query(
                patch_fingerprint=patch.patch_fingerprint,
                limit=1,
            )
            if duplicate:
                issues.append(
                    _issue(
                        "duplicate_patch",
                        "An equivalent Candidate patch already exists.",
                    )
                )
            if issues:
                revisions.append(
                    CandidateRevision(
                        round_no=1,
                        patch=patch,
                        deterministic_issues=issues,
                        changed_from_previous=changed,
                        generator_input_tokens=generator_response.input_tokens,
                        generator_output_tokens=generator_response.output_tokens,
                    )
                )
                final_status = "reject"
                final_reasons = [item.message for item in issues]
            else:
                raw_verification, verifier_response = await self._call_json(
                    self.verifier_llm,
                    _VERIFIER_SYSTEM,
                    {
                        "capsule": parent.capsule.model_dump(mode="json"),
                        "candidate_patch": patch.model_dump(mode="json"),
                        "active_verifier_assertions": assertions,
                    },
                    max_tokens=self.config.verifier_max_tokens,
                    meter=meter,
                )
                verification = _verification_from_payload(
                    raw_verification, verifier_response
                )
                revisions.append(
                    CandidateRevision(
                        round_no=1,
                        patch=patch,
                        verification=verification,
                        changed_from_previous=True,
                        generator_input_tokens=generator_response.input_tokens,
                        generator_output_tokens=generator_response.output_tokens,
                    )
                )
                if verification.decision == "accept_for_replay":
                    final_status = "accept_for_replay"
                elif verification.decision == "reject":
                    final_status = "reject"
                else:
                    final_status = "hold"
                final_reasons = [
                    verification.summary,
                    *[item.message for item in verification.issues],
                ]
        except (CandidateTokenBudgetExceeded, LLMJSONError, ValidationError, ValueError) as exc:
            final_status = "hold"
            final_reasons = [f"{type(exc).__name__}: {exc}"]
        record = self._terminal_record(
            episode=episode,
            candidate_id=candidate_id,
            status=final_status,
            reasons=final_reasons,
            capsule=parent.capsule,
            revisions=revisions,
            final_patch=final_patch,
            meter=meter,
            parent_candidate_id=parent.candidate_id,
            source_verifier_gap_ids=[gap.gap_id],
        )
        return self._persist(record)
