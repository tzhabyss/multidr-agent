"""Isolated, immutable storage for phase-one Evolution Episodes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .models import CandidateRecord, EvolutionEpisode


class EpisodeStoreConflict(RuntimeError):
    pass


class CandidateStoreConflict(RuntimeError):
    pass


class EvolutionEpisodeStore:
    """Persist Episodes outside ``skills/`` so the Skill loader cannot see them."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.episodes_root = self.root / "episodes"
        self.candidates_root = self.root / "candidates"
        self.releases_root = self.root / "releases"

    @staticmethod
    def _safe_run_id(run_id: str) -> str:
        if not run_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in run_id
        ):
            raise ValueError("run_id contains unsafe characters")
        return run_id

    def path_for_run(self, run_id: str) -> Path:
        return self.episodes_root / f"{self._safe_run_id(run_id)}.json"

    def write_run(
        self, run_id: str, episodes: Iterable[EvolutionEpisode]
    ) -> Path:
        target = self.path_for_run(run_id)
        values = [item.model_dump(mode="json") for item in episodes]
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "episodes": values,
        }
        data = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() == data:
                return target
            raise EpisodeStoreConflict(
                f"Evolution Episodes for run {run_id!r} are immutable"
            )
        descriptor, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return target

    def read_run(self, run_id: str) -> list[EvolutionEpisode]:
        value = json.loads(self.path_for_run(run_id).read_text(encoding="utf-8"))
        return [EvolutionEpisode.model_validate(item) for item in value["episodes"]]

    def query(
        self,
        *,
        target_skill_id: str = "",
        task_id: str = "",
        trigger_type: str = "",
        limit: int = 100,
    ) -> list[EvolutionEpisode]:
        """Retrieve review/replay inputs without reading original conversations."""

        if limit < 1:
            return []
        matches: list[EvolutionEpisode] = []
        if not self.episodes_root.exists():
            return matches
        for path in sorted(self.episodes_root.glob("*.json"), reverse=True):
            value = json.loads(path.read_text(encoding="utf-8"))
            for raw in value.get("episodes", []):
                item = EvolutionEpisode.model_validate(raw)
                if target_skill_id and item.target_skill_id != target_skill_id:
                    continue
                if task_id and item.task_id != task_id:
                    continue
                if trigger_type and trigger_type not in item.trigger_types:
                    continue
                matches.append(item)
                if len(matches) >= limit:
                    return matches
        return matches

    def get_episode(self, episode_id: str) -> EvolutionEpisode | None:
        if not self.episodes_root.exists():
            return None
        for path in sorted(self.episodes_root.glob("*.json"), reverse=True):
            value = json.loads(path.read_text(encoding="utf-8"))
            for raw in value.get("episodes", []):
                if str(raw.get("episode_id") or "") == episode_id:
                    return EvolutionEpisode.model_validate(raw)
        return None


class CandidateStore:
    """Immutable Candidate records, isolated from the Stable Skill loader."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.candidates_root = self.root / "candidates"

    @staticmethod
    def _safe_candidate_id(candidate_id: str) -> str:
        if not candidate_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in candidate_id
        ):
            raise ValueError("candidate_id contains unsafe characters")
        return candidate_id

    def path_for_candidate(self, candidate_id: str) -> Path:
        return self.candidates_root / f"{self._safe_candidate_id(candidate_id)}.json"

    def write(self, record: CandidateRecord) -> Path:
        target = self.path_for_candidate(record.candidate_id)
        data = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() == data:
                return target
            raise CandidateStoreConflict(
                f"Candidate record {record.candidate_id!r} is immutable"
            )
        descriptor, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return target

    def read(self, candidate_id: str) -> CandidateRecord:
        value = json.loads(
            self.path_for_candidate(candidate_id).read_text(encoding="utf-8")
        )
        return CandidateRecord.model_validate(value)

    def query(
        self,
        *,
        source_episode_id: str = "",
        target_skill_id: str = "",
        status: str = "",
        patch_fingerprint: str = "",
        limit: int = 100,
    ) -> list[CandidateRecord]:
        if limit < 1 or not self.candidates_root.exists():
            return []
        matches: list[CandidateRecord] = []
        for path in sorted(self.candidates_root.glob("*.json"), reverse=True):
            item = CandidateRecord.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if source_episode_id and item.source_episode_id != source_episode_id:
                continue
            if target_skill_id and item.target_skill_id != target_skill_id:
                continue
            if status and item.status != status:
                continue
            if patch_fingerprint and (
                item.final_patch is None
                or item.final_patch.patch_fingerprint != patch_fingerprint
            ):
                continue
            matches.append(item)
            if len(matches) >= limit:
                break
        return matches

    def find_for_episode(self, episode_id: str) -> CandidateRecord | None:
        matches = self.query(source_episode_id=episode_id, limit=1)
        return matches[0] if matches else None
