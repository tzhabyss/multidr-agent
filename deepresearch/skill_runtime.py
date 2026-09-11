"""Instruction-only activation, immutable run snapshots, explicit rollback."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .skill_loader import _parse_frontmatter, discover_skills


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class SkillRuntimeManager:
    def __init__(self, registry):
        self.registry = registry
        self.previous = []
        self.manifests = {}
        self.executor_hash = self._executor_hash()
        for key, item in list(registry._active_skills.items()):
            manifest = Path(item.source_path) / "skill.json"
            self.manifests[key] = digest(manifest.read_text()) if manifest.is_file() else ""
            body = item.instructions_loader() if not item.instructions_loaded and item.instructions_loader else item.instructions
            registry._active_skills[key] = replace(
                item, instructions_loader=(lambda text=body: text) if not item.instructions_loaded else None
            )

    def _executor_hash(self):
        root = Path(__file__).parent
        return digest("".join(str(p.relative_to(root)) + digest(p.read_text())
                              for p in sorted(root.rglob("*.py"))))

    def snapshot(self):
        items = {}
        for key, item in self.registry._skills.items():
            body = item.instructions_loader() if not item.instructions_loaded and item.instructions_loader else item.instructions
            items[key] = {
                "version": item.skill_version, "description": item.description,
                "instructions": body, "content_hash": digest(body),
                "manifest_hash": self.manifests[key],
                "roles": sorted(item.roles), "profiles": sorted(item.profiles),
                "parameters": item.parameters, "source": item.source_path,
            }
        payload = {"skills": items, "executor_hash": self.executor_hash}
        return {**payload, "snapshot_id": digest(json.dumps(payload, sort_keys=True))}

    @contextmanager
    def bind(self, snapshot):
        payload = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
        if digest(json.dumps(payload, sort_keys=True)) != snapshot["snapshot_id"]:
            raise ValueError("Skill snapshot integrity check failed")
        if snapshot["executor_hash"] != self._executor_hash():
            raise ValueError("Executor changed; restore the original code before resuming")
        frozen = {}
        for key, row in snapshot["skills"].items():
            current = self.registry._active_skills.get(key)
            if current is None:
                raise ValueError(f"Snapshot executor unavailable: {key}")
            if row["roles"] != sorted(current.roles) or row["profiles"] != sorted(current.profiles) or row["parameters"] != current.parameters:
                raise ValueError(f"Skill permissions/schema changed: {key}")
            manifest = Path(current.source_path) / "skill.json"
            if row["manifest_hash"] and (not manifest.is_file() or digest(manifest.read_text()) != row["manifest_hash"]):
                raise ValueError(f"Skill executor configuration changed: {key}")
            if digest(row["instructions"]) != row["content_hash"]:
                raise ValueError(f"Corrupt Skill contents: {key}")
            frozen[key] = replace(current, instructions="",
                                  description=row["description"], skill_version=row["version"],
                                  instructions_loader=lambda text=row["instructions"]: text, instructions_loaded=False)
        token = self.registry._snapshot_skills.set(frozen)
        try:
            yield
        finally:
            self.registry._snapshot_skills.reset(token)

    def reload(self, *, approved=False):
        """Reload installed instructions only, never evolution candidate directories."""
        updated = dict(self.registry._active_skills)
        errors, changed = {}, []
        pending_manifests = dict(self.manifests)
        for key, item in updated.copy().items():
            if not item.instructions_path:
                continue
            try:
                manifest = Path(item.source_path) / "skill.json"
                if self.manifests[key] and digest(manifest.read_text()) != self.manifests[key]:
                    raise ValueError("Executor manifest updates require service restart")
                metadata, body = _parse_frontmatter(Path(item.instructions_path))
                if metadata["name"] != key or not body:
                    raise ValueError("Skill name changed or instructions empty")
                if body != item.load_instructions() or metadata["description"] != item.description:
                    changed.append(key)
                    updated[key] = replace(item, instructions=body, instructions_loader=None,
                                           instructions_loaded=True, description=metadata["description"])
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
        roots = {Path(item.source_path).parent for item in updated.values() if item.instructions_path}
        for root in roots:
            for directory in sorted(root.iterdir()):
                if not directory.is_dir() or directory.name in updated:
                    continue
                try:
                    additions = discover_skills(root.parent, lazy=True, only={directory.name})
                    for item in additions:
                        if item.executable:
                            raise ValueError("New executors require service restart")
                        if any(existing.tool_name == item.tool_name for existing in updated.values()):
                            raise ValueError("Skill tool name already registered")
                        self.registry._validate_arguments_schema(item.parameters)
                        body = item.load_instructions()
                        if not body:
                            raise ValueError("Empty instructions")
                        updated[item.skill_id] = item
                        changed.append(item.skill_id)
                        pending_manifests[item.skill_id] = digest((directory / "skill.json").read_text())
                except Exception as exc:
                    errors[directory.name] = f"{type(exc).__name__}: {exc}"
        if approved and changed:
            self.previous.append(dict(self.registry._active_skills))
            self.previous = self.previous[-10:]
            self.registry._active_skills = updated
            self.manifests = pending_manifests
        return {"activated": approved and bool(changed), "changed": changed, "errors": errors}

    def rollback(self):
        if not self.previous:
            raise ValueError("No previous Skill activation")
        self.registry._active_skills = self.previous.pop()
