"""Immutable stores for replay, verifier gaps, suites, and activation records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .verifier_models import (
    CandidateReplayResult,
    VerifierGap,
    VerifierSuite,
    VerifierSuiteRelease,
)


class VerifierEvolutionStoreConflict(RuntimeError):
    pass


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _safe_component(value: str, label: str) -> str:
    if not value or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for character in value
    ):
        raise ValueError(f"{label} contains unsafe characters")
    return value


def _write_immutable(path: Path, value: BaseModel) -> Path:
    data = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise VerifierEvolutionStoreConflict(f"Record {path.name!r} is immutable")
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
    return path


def _read(path: Path, model: type[_ModelT]) -> _ModelT:
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


class VerifierEvolutionStore:
    """Keep experimental verifier state outside the Stable Skill directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.replays_root = self.root / "replays"
        self.gaps_root = self.root / "verifier-gaps"
        self.suites_root = self.root / "verifier-suites"
        self.releases_root = self.root / "verifier-releases"

    def replay_path(self, replay_id: str) -> Path:
        return self.replays_root / f"{_safe_component(replay_id, 'replay_id')}.json"

    def write_replay(self, value: CandidateReplayResult) -> Path:
        return _write_immutable(self.replay_path(value.replay_id), value)

    def read_replay(self, replay_id: str) -> CandidateReplayResult | None:
        path = self.replay_path(replay_id)
        return _read(path, CandidateReplayResult) if path.is_file() else None

    def gap_path(self, gap_id: str) -> Path:
        return self.gaps_root / f"{_safe_component(gap_id, 'gap_id')}.json"

    def write_gap(self, value: VerifierGap) -> Path:
        return _write_immutable(self.gap_path(value.gap_id), value)

    def read_gap(self, gap_id: str) -> VerifierGap | None:
        path = self.gap_path(gap_id)
        return _read(path, VerifierGap) if path.is_file() else None

    def suite_path(self, suite_id: str, version: str) -> Path:
        return (
            self.suites_root
            / _safe_component(suite_id, "verifier_suite_id")
            / f"{_safe_component(version, 'version')}.json"
        )

    def write_suite(self, value: VerifierSuite) -> Path:
        return _write_immutable(
            self.suite_path(value.verifier_suite_id, value.version), value
        )

    def read_suite(self, suite_id: str, version: str) -> VerifierSuite | None:
        path = self.suite_path(suite_id, version)
        return _read(path, VerifierSuite) if path.is_file() else None

    def find_suite_for_gap(self, gap_id: str) -> VerifierSuite | None:
        if not self.suites_root.exists():
            return None
        for path in sorted(self.suites_root.glob("*/*.json"), reverse=True):
            value = _read(path, VerifierSuite)
            if gap_id in value.source_gap_ids:
                return value
        return None

    def pending_suite(
        self, target_skill_id: str, parent_version: str
    ) -> VerifierSuite | None:
        """Return an unactivated child of the current parent, if one exists."""

        if not self.suites_root.exists():
            return None
        active = self.active_release(target_skill_id)
        active_version = active.version if active is not None else "1.0.0"
        for path in sorted(self.suites_root.glob("*/*.json"), reverse=True):
            value = _read(path, VerifierSuite)
            if value.target_skill_id != target_skill_id:
                continue
            if value.parent_version != parent_version:
                continue
            if value.version != active_version:
                return value
        return None

    def release_path(self, release_id: str) -> Path:
        return self.releases_root / f"{_safe_component(release_id, 'release_id')}.json"

    def write_release(self, value: VerifierSuiteRelease) -> Path:
        return _write_immutable(self.release_path(value.release_id), value)

    def active_release(self, target_skill_id: str) -> VerifierSuiteRelease | None:
        if not self.releases_root.exists():
            return None
        matches: list[VerifierSuiteRelease] = []
        for path in self.releases_root.glob("*.json"):
            value = _read(path, VerifierSuiteRelease)
            if value.target_skill_id == target_skill_id:
                matches.append(value)
        if not matches:
            return None
        return max(matches, key=lambda item: (_version_tuple(item.version), item.created_at))

    def active_suite(self, target_skill_id: str) -> VerifierSuite | None:
        release = self.active_release(target_skill_id)
        if release is None:
            return None
        return self.read_suite(release.verifier_suite_id, release.version)


def _version_tuple(value: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = value.split(".", 2)
        return int(major), int(minor), int(patch)
    except (TypeError, ValueError):
        return 0, 0, 0
