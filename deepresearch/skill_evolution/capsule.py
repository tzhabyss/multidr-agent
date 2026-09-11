"""Build bounded, local input capsules for CoEvoSkills-Lite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ids import make_id
from ..skill_registry import SkillRegistry
from .models import (
    CandidateEvidenceExcerpt,
    CandidateInputCapsule,
    EvolutionEpisode,
)
from .store import EvolutionEpisodeStore


_EXPECTED_OUTCOMES: dict[str, list[str]] = {
    "consecutive_no_evidence_gain": [
        "下一次查询必须改变实体锚点、关系角度、来源家族、语言或时间约束。",
        "避免继续执行与低增益查询语义等价的检索。",
    ],
    "duplicate_query": ["降低局部查询重复率，同时保持目标研究范围不变。"],
    "repeated_source_domain": ["切换到新的、已授权的来源家族，优先第一方来源。"],
    "source_family_repeated_failure": ["在有界次数内切换既有抓取回退路线。"],
    "fetch_route_failure": ["对可恢复抓取错误使用既有安全回退，不绕过访问挑战。"],
    "extraction_without_binding": ["只在原文直接支持时形成 Evidence Binding。"],
    "skill_execution_failure": ["避免重复触发相同 Skill 调用错误。"],
    "weak_or_rejected_claim": ["收窄 Claim，使其完全落在直接证据范围内。"],
}


class CandidateCapsuleError(RuntimeError):
    pass


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _find_record(value: Any, record_id: str) -> Any | None:
    if isinstance(value, dict):
        if record_id and record_id in {str(item) for item in value.values()}:
            return value
        for child in value.values():
            found = _find_record(child, record_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_record(child, record_id)
            if found is not None:
                return found
    return None


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: max(0, limit - 20)] + "…[truncated]", True


class CandidateCapsuleBuilder:
    """Resolve only Episode-linked records and the target Skill body."""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        episode_store: EvolutionEpisodeStore,
        artifact_root: str | Path,
        max_similar_episodes: int,
        max_capsule_chars: int,
        parent_skill_excerpt_chars: int,
        artifact_excerpt_chars: int,
        max_patch_operations: int,
        max_patch_chars: int,
    ) -> None:
        self.registry = registry
        self.episode_store = episode_store
        self.artifact_root = Path(artifact_root).resolve()
        self.max_similar_episodes = max_similar_episodes
        self.max_capsule_chars = max_capsule_chars
        self.parent_skill_excerpt_chars = parent_skill_excerpt_chars
        self.artifact_excerpt_chars = artifact_excerpt_chars
        self.max_patch_operations = max_patch_operations
        self.max_patch_chars = max_patch_chars

    def _artifact_path(self, episode: EvolutionEpisode, relative: str) -> Path:
        run_root = (self.artifact_root / episode.run_id).resolve()
        target = (run_root / relative).resolve()
        if target != run_root and run_root not in target.parents:
            raise CandidateCapsuleError(
                f"Episode evidence path escapes its run artifact: {relative}"
            )
        return target

    def _evidence_excerpts(
        self, episode: EvolutionEpisode
    ) -> list[CandidateEvidenceExcerpt]:
        excerpts: list[CandidateEvidenceExcerpt] = []
        for reference in episode.evidence_refs:
            path = self._artifact_path(episode, reference.artifact_path)
            if not path.exists() or not path.is_file():
                raise CandidateCapsuleError(
                    f"Episode evidence artifact is missing: {reference.artifact_path}"
                )
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CandidateCapsuleError(
                    f"Episode evidence artifact is unreadable: {reference.artifact_path}"
                ) from exc
            selected = (
                _find_record(value, reference.record_id)
                if reference.record_id
                else value
            )
            if selected is None:
                raise CandidateCapsuleError(
                    "Episode evidence record is missing: "
                    f"{reference.artifact_path}#{reference.record_id}"
                )
            content, truncated = _bounded(
                _json_text(selected), self.artifact_excerpt_chars
            )
            excerpts.append(
                CandidateEvidenceExcerpt(
                    artifact_path=reference.artifact_path,
                    record_type=reference.record_type,
                    record_id=reference.record_id,
                    content=content,
                    truncated=truncated,
                )
            )
        return excerpts

    def _similar_episodes(self, episode: EvolutionEpisode) -> list[dict[str, Any]]:
        if self.max_similar_episodes <= 0:
            return []
        matches = self.episode_store.query(
            target_skill_id=episode.target_skill_id,
            limit=self.max_similar_episodes + 1,
        )
        return [
            {
                "episode_id": item.episode_id,
                "trigger_types": item.trigger_types,
                "scope": item.scope,
                "summary": item.local_context.get("summary", ""),
                "confidence": item.confidence,
            }
            for item in matches
            if item.episode_id != episode.episode_id
        ][: self.max_similar_episodes]

    @staticmethod
    def _measure(payload: dict[str, Any]) -> int:
        return len(_json_text(payload))

    def build(self, episode: EvolutionEpisode) -> CandidateInputCapsule:
        definition = self.registry.get(episode.target_skill_id)
        if definition is None:
            raise CandidateCapsuleError(
                f"Target Skill is not registered: {episode.target_skill_id}"
            )
        if definition.skill_version != episode.target_skill_version:
            raise CandidateCapsuleError(
                "Episode parent version does not match Stable Skill: "
                f"{episode.target_skill_version} != {definition.skill_version}"
            )
        parent_excerpt, parent_truncated = _bounded(
            definition.load_instructions(), self.parent_skill_excerpt_chars
        )
        evidence = self._evidence_excerpts(episode)
        trigger_type = episode.trigger_types[0]
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "capsule_id": make_id("candidate-capsule", episode.episode_id),
            "source_episode_id": episode.episode_id,
            "run_id": episode.run_id,
            "task_id": episode.task_id,
            "primary_trigger_type": trigger_type,
            "trigger_summary": str(episode.local_context.get("summary") or ""),
            "target_skill_id": episode.target_skill_id,
            "parent_skill_version": episode.target_skill_version,
            "runtime_layer": episode.runtime_layer,
            "capability_granularity": episode.capability_granularity,
            "candidate_policy": episode.candidate_policy,
            "scope": episode.scope,
            "local_context": episode.local_context,
            "local_actions": episode.actions,
            "local_result": episode.local_result,
            "expected_local_outcomes": _EXPECTED_OUTCOMES.get(trigger_type, []),
            "parent_skill_excerpt": parent_excerpt,
            "parent_skill_source_path": definition.instructions_path,
            "evidence_excerpts": [item.model_dump(mode="json") for item in evidence],
            "similar_episodes": self._similar_episodes(episode),
            "allowed_boundary": {
                "allowed_operations": ["add", "delete", "replace"],
                "max_operations": self.max_patch_operations,
                "max_patch_chars": self.max_patch_chars,
                "target_skill_only": episode.target_skill_id,
                "permission_expansion_allowed": False,
                "new_tools_allowed": False,
                "citation_or_evidence_bypass_allowed": False,
                "external_writes_allowed": False,
            },
            "total_chars": 0,
            "truncated": parent_truncated or any(item.truncated for item in evidence),
        }
        payload["total_chars"] = self._measure(payload)
        if payload["total_chars"] > self.max_capsule_chars:
            payload["similar_episodes"] = []
            payload["truncated"] = True
            payload["total_chars"] = self._measure(payload)
        if payload["total_chars"] > self.max_capsule_chars:
            overflow = payload["total_chars"] - self.max_capsule_chars
            reduced, _ = _bounded(
                payload["parent_skill_excerpt"],
                max(500, len(payload["parent_skill_excerpt"]) - overflow - 64),
            )
            payload["parent_skill_excerpt"] = reduced
            payload["total_chars"] = self._measure(payload)
        if payload["total_chars"] > self.max_capsule_chars:
            raise CandidateCapsuleError(
                f"Candidate capsule exceeds {self.max_capsule_chars} characters"
            )
        return CandidateInputCapsule.model_validate(payload)
