"""Materialize a Skill Candidate into an isolated replay workspace."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..ids import stable_hash
from ..skill_registry import SkillRegistry
from .models import CandidatePatch, CandidateRecord


class CandidateMaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    descriptor, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _split_skill_file(content: str) -> tuple[str, str]:
    if not content.startswith("---"):
        raise CandidateMaterializationError("Stable SKILL.md has no frontmatter")
    parts = content.split("---", 2)
    if len(parts) != 3:
        raise CandidateMaterializationError("Stable SKILL.md frontmatter is not closed")
    return parts[1], parts[2].strip()


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    occurrences = text.count(old)
    if occurrences != 1:
        raise CandidateMaterializationError(
            f"{label} must match exactly once; found {occurrences}"
        )
    return text.replace(old, new, 1)


def apply_candidate_patch(parent_text: str, patch: CandidatePatch) -> str:
    """Apply exact, bounded operations without interpreting Markdown."""

    updated = parent_text
    for operation in patch.operations:
        if operation.kind == "add":
            if not operation.anchor:
                raise CandidateMaterializationError("Add operation has no anchor")
            replacement = operation.anchor + "\n\n" + operation.new_text.strip()
            updated = _replace_once(
                updated,
                operation.anchor,
                replacement,
                label=f"add anchor {operation.operation_id}",
            )
        elif operation.kind == "delete":
            updated = _replace_once(
                updated,
                operation.old_text,
                "",
                label=f"delete text {operation.operation_id}",
            )
        elif operation.kind == "replace":
            updated = _replace_once(
                updated,
                operation.old_text,
                operation.new_text,
                label=f"replace text {operation.operation_id}",
            )
        else:  # pragma: no cover - validated by CandidatePatch
            raise CandidateMaterializationError(
                f"Unsupported operation kind: {operation.kind}"
            )
    if updated == parent_text:
        raise CandidateMaterializationError("Candidate patch made no change")
    return updated


@dataclass(frozen=True)
class MaterializedCandidate:
    replay_root: Path
    stable_project_root: Path
    candidate_project_root: Path
    stable_skill_path: Path
    candidate_skill_path: Path
    manifest_path: Path
    stable_sha256: str
    candidate_sha256: str
    candidate_version: str


class CandidateSkillMaterializer:
    """Copy one Skill into Stable/Candidate sandboxes and patch only Candidate."""

    def __init__(self, *, registry: SkillRegistry, replay_root: str | Path) -> None:
        self.registry = registry
        self.replay_root = Path(replay_root).resolve()

    def materialize(
        self, candidate: CandidateRecord, *, replay_id: str
    ) -> MaterializedCandidate:
        if candidate.status != "accept_for_replay" or candidate.final_patch is None:
            raise CandidateMaterializationError(
                "Only an accept_for_replay Candidate with a final patch may be materialized"
            )
        if candidate.stable_skill_mutation_enabled or candidate.production_loading_enabled:
            raise CandidateMaterializationError(
                "Replay Candidate must not enable Stable mutation or production loading"
            )
        patch = candidate.final_patch
        if patch.requested_tools or patch.requested_permissions or patch.gate_bypasses:
            raise CandidateMaterializationError(
                "Replay Candidate requests tools, permissions, or a gate bypass"
            )
        if patch.target_skill_id != candidate.target_skill_id:
            raise CandidateMaterializationError("Candidate target Skill mismatch")
        definition = self.registry.get(candidate.target_skill_id)
        if definition is None:
            raise CandidateMaterializationError(
                f"Stable Skill not found: {candidate.target_skill_id}"
            )
        if definition.skill_version != candidate.parent_skill_version:
            raise CandidateMaterializationError(
                "Stable Skill version no longer matches Candidate parent version"
            )
        source_dir = Path(definition.source_path).resolve()
        skill_file = source_dir / "SKILL.md"
        manifest_file = source_dir / "skill.json"
        if not skill_file.is_file() or not manifest_file.is_file():
            raise CandidateMaterializationError("Stable Skill files are incomplete")

        parent_content = skill_file.read_text(encoding="utf-8")
        frontmatter_text, parent_body = _split_skill_file(parent_content)
        metadata = yaml.safe_load(frontmatter_text) or {}
        if str(metadata.get("name") or "") != candidate.target_skill_id:
            raise CandidateMaterializationError("Stable Skill frontmatter name mismatch")
        patched_body = apply_candidate_patch(parent_body, patch)
        candidate_content = f"---{frontmatter_text}---\n\n{patched_body.rstrip()}\n"
        expected_stable_sha256 = hashlib.sha256(skill_file.read_bytes()).hexdigest()
        expected_candidate_sha256 = hashlib.sha256(
            candidate_content.encode("utf-8")
        ).hexdigest()
        candidate_version = (
            f"{candidate.parent_skill_version}+candidate."
            f"{stable_hash(candidate.candidate_id, patch.patch_fingerprint)[:8]}"
        )

        replay_dir = (self.replay_root / replay_id).resolve()
        if replay_dir == self.replay_root or self.replay_root not in replay_dir.parents:
            raise CandidateMaterializationError("Replay path escapes replay root")
        stable_project = replay_dir / "stable"
        candidate_project = replay_dir / "candidate"
        stable_dir = stable_project / "skills" / candidate.target_skill_id
        candidate_dir = candidate_project / "skills" / candidate.target_skill_id
        candidate_skill_file = candidate_dir / "SKILL.md"
        manifest_path = replay_dir / "materialization.json"
        if replay_dir.exists():
            try:
                recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CandidateMaterializationError(
                    f"Existing replay workspace is incomplete: {replay_dir}"
                ) from exc
            checks = {
                "replay_id": replay_id,
                "candidate_id": candidate.candidate_id,
                "target_skill_id": candidate.target_skill_id,
                "parent_skill_version": candidate.parent_skill_version,
                "candidate_version": candidate_version,
                "patch_fingerprint": patch.patch_fingerprint,
                "stable_sha256": expected_stable_sha256,
                "candidate_sha256": expected_candidate_sha256,
            }
            if any(recorded.get(key) != value for key, value in checks.items()):
                raise CandidateMaterializationError(
                    f"Existing replay workspace does not match Candidate: {replay_dir}"
                )
            if (
                not (stable_dir / "SKILL.md").is_file()
                or not candidate_skill_file.is_file()
                or _sha256(stable_dir / "SKILL.md") != expected_stable_sha256
                or _sha256(candidate_skill_file) != expected_candidate_sha256
            ):
                raise CandidateMaterializationError(
                    f"Existing replay workspace failed integrity checks: {replay_dir}"
                )
            return MaterializedCandidate(
                replay_root=replay_dir,
                stable_project_root=stable_project,
                candidate_project_root=candidate_project,
                stable_skill_path=stable_dir / "SKILL.md",
                candidate_skill_path=candidate_skill_file,
                manifest_path=manifest_path,
                stable_sha256=expected_stable_sha256,
                candidate_sha256=expected_candidate_sha256,
                candidate_version=candidate_version,
            )

        self.replay_root.mkdir(parents=True, exist_ok=True)
        build_dir = Path(
            tempfile.mkdtemp(
                prefix=f".build-{stable_hash(replay_id)[:8]}-",
                dir=self.replay_root,
            )
        )
        build_stable_dir = build_dir / "stable" / "skills" / candidate.target_skill_id
        build_candidate_dir = (
            build_dir / "candidate" / "skills" / candidate.target_skill_id
        )
        try:
            build_stable_dir.mkdir(parents=True)
            build_candidate_dir.mkdir(parents=True)
            shutil.copy2(skill_file, build_stable_dir / "SKILL.md")
            shutil.copy2(manifest_file, build_stable_dir / "skill.json")
            shutil.copy2(skill_file, build_candidate_dir / "SKILL.md")
            (build_candidate_dir / "SKILL.md").write_text(
                candidate_content,
                encoding="utf-8",
            )
            skill_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            skill_manifest["version"] = candidate_version
            _atomic_json(build_candidate_dir / "skill.json", skill_manifest)
            if (
                _sha256(build_stable_dir / "SKILL.md") != expected_stable_sha256
                or _sha256(build_candidate_dir / "SKILL.md")
                != expected_candidate_sha256
            ):
                raise CandidateMaterializationError(
                    "Candidate replay build failed integrity checks"
                )
            _atomic_json(build_dir / "materialization.json", {
                "schema_version": "1.0",
                "replay_id": replay_id,
                "candidate_id": candidate.candidate_id,
                "target_skill_id": candidate.target_skill_id,
                "parent_skill_version": candidate.parent_skill_version,
                "candidate_version": candidate_version,
                "patch_fingerprint": patch.patch_fingerprint,
                "stable_skill_path": str(stable_dir / "SKILL.md"),
                "candidate_skill_path": str(candidate_skill_file),
                "stable_sha256": expected_stable_sha256,
                "candidate_sha256": expected_candidate_sha256,
                "stable_skill_mutated": False,
                "production_loading_enabled": False,
            })
            os.replace(build_dir, replay_dir)
        finally:
            if build_dir.exists():
                shutil.rmtree(build_dir)

        stable_sha256 = _sha256(stable_dir / "SKILL.md")
        candidate_sha256 = _sha256(candidate_skill_file)
        materialization_manifest = {
            "schema_version": "1.0",
            "replay_id": replay_id,
            "candidate_id": candidate.candidate_id,
            "target_skill_id": candidate.target_skill_id,
            "parent_skill_version": candidate.parent_skill_version,
            "candidate_version": candidate_version,
            "patch_fingerprint": patch.patch_fingerprint,
            "stable_skill_path": str(stable_dir / "SKILL.md"),
            "candidate_skill_path": str(candidate_skill_file),
            "stable_sha256": stable_sha256,
            "candidate_sha256": candidate_sha256,
            "stable_skill_mutated": False,
            "production_loading_enabled": False,
        }
        # The manifest was written inside the temporary directory before the
        # atomic rename. Keep this local value for the returned audit record.
        if json.loads(manifest_path.read_text(encoding="utf-8")) != materialization_manifest:
            raise CandidateMaterializationError("Materialization manifest mismatch")
        return MaterializedCandidate(
            replay_root=replay_dir,
            stable_project_root=stable_project,
            candidate_project_root=candidate_project,
            stable_skill_path=stable_dir / "SKILL.md",
            candidate_skill_path=candidate_skill_file,
            manifest_path=manifest_path,
            stable_sha256=stable_sha256,
            candidate_sha256=candidate_sha256,
            candidate_version=candidate_version,
        )
