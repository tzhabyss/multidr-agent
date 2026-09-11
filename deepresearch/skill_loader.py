"""Discover SKILL.md folders and bind declarative runtime entrypoints."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .skill_registry import SkillDefinition, SkillError, SkillRegistry


def _env_flag(name: str | None) -> bool:
    if not name:
        return False
    return os.getenv(name, "").strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_frontmatter(path: Path, *, include_body: bool = True) -> tuple[dict[str, Any], str]:
    if include_body:
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            raise SkillError(f"Skill file has no YAML frontmatter: {path}")
        parts = content.split("---", 2)
        if len(parts) != 3:
            raise SkillError(f"Skill frontmatter is not closed: {path}")
        frontmatter, body = parts[1], parts[2].strip()
    else:
        # Read only through the closing YAML delimiter.  The potentially large
        # instruction body is not read until a role explicitly selects it.
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if first.strip() != "---":
                raise SkillError(f"Skill file has no YAML frontmatter: {path}")
            lines: list[str] = []
            for line in handle:
                if line.strip() == "---":
                    break
                lines.append(line)
            else:
                raise SkillError(f"Skill frontmatter is not closed: {path}")
        frontmatter, body = "".join(lines), ""
    metadata = yaml.safe_load(frontmatter) or {}
    if not isinstance(metadata, dict) or not metadata.get("name") or not metadata.get(
        "description"
    ):
        raise SkillError(f"Skill requires name and description: {path}")
    return metadata, body


def _load_instruction_body(path: Path) -> str:
    return _parse_frontmatter(path, include_body=True)[1]


def _resolve_handler(
    config: dict[str, Any], override: Callable[..., Any] | None
) -> Callable[..., Any] | None:
    if override is not None:
        return override
    entrypoint = config.get("entrypoint")
    if not entrypoint:
        return None
    module_name, separator, attribute_name = str(entrypoint).partition(":")
    if not separator:
        raise SkillError(f"Invalid skill entrypoint: {entrypoint}")
    target = getattr(importlib.import_module(module_name), attribute_name)
    instance = target(**config.get("init_kwargs", {})) if inspect.isclass(target) else target
    method_name = config.get("method")
    return getattr(instance, method_name) if method_name else instance


def discover_skills(
    project_root: str | Path,
    *,
    runtime_overrides: dict[str, Callable[..., Any]] | None = None,
    lazy: bool = False,
    only: set[str] | None = None,
) -> list[SkillDefinition]:
    root = Path(project_root).resolve() / "skills"
    overrides = runtime_overrides or {}
    if not root.exists():
        raise SkillError(f"Skills directory does not exist: {root}")
    definitions: list[SkillDefinition] = []
    for skill_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        if only is not None and skill_dir.name not in only:
            continue
        skill_path = skill_dir / "SKILL.md"
        manifest_path = skill_dir / "skill.json"
        if not skill_path.exists() or not manifest_path.exists():
            continue
        metadata, instructions = _parse_frontmatter(
            skill_path, include_body=not lazy
        )
        config = json.loads(manifest_path.read_text(encoding="utf-8"))
        skill_id = str(metadata["name"])
        if skill_dir.name != skill_id:
            raise SkillError(
                f"Skill folder {skill_dir.name!r} must match metadata name {skill_id!r}"
            )
        required_env = config.get("enabled_env")
        disabled_env = config.get("disabled_env")
        enabled = (not required_env or _env_flag(str(required_env))) and not _env_flag(
            str(disabled_env) if disabled_env else None
        )
        override = overrides.get(skill_id) or overrides.get(skill_id.replace("-", "_"))
        handler = override
        handler_loader = None
        if override is None and enabled and config.get("entrypoint"):
            if lazy:
                handler_loader = lambda config=dict(config): _resolve_handler(config, None)
            else:
                handler = _resolve_handler(config, None)
        elif not lazy and enabled:
            handler = _resolve_handler(config, override)
        definitions.append(
            SkillDefinition(
                skill_id=skill_id,
                description=str(metadata["description"]).strip(),
                instructions=instructions,
                roles=frozenset(config.get("roles", [])),
                profiles=frozenset(config.get("profiles", [])),
                parameters=config.get("parameters", {}),
                handler=handler,
                fallback_skills=tuple(config.get("fallback_skills", [])),
                timeout_seconds=float(config.get("timeout_seconds", 30)),
                tags=frozenset(config.get("tags", [])),
                source_path=str(skill_dir),
                enabled=enabled or override is not None,
                result_kind=str(config.get("result_kind", "search_hits")),
                skill_layer=str(config.get("skill_layer", "strategy")),
                capability_granularity=str(
                    config.get("capability_granularity", "functional")
                ),
                skill_version=str(config.get("version", "1.0.0")),
                applicable_entity_types=frozenset(
                    config.get("applicable_entity_types", [])
                ),
                applicable_source_families=frozenset(
                    config.get("applicable_source_families", [])
                ),
                required_tools=frozenset(config.get("required_tools", [])),
                output_artifact_roles=frozenset(
                    config.get("output_artifact_roles", [])
                ),
                failure_signatures=frozenset(
                    config.get("failure_signatures", [])
                ),
                cost_class=str(config.get("cost_class", "medium")),
                instructions_path=str(skill_path),
                instructions_loader=(
                    (lambda path=skill_path: _load_instruction_body(path))
                    if lazy
                    else None
                ),
                handler_loader=handler_loader,
                instructions_loaded=not lazy,
                handler_loaded=(handler is not None or not bool(handler_loader)),
            )
        )
    return definitions


def load_skill_registry(
    project_root: str | Path,
    *,
    runtime_overrides: dict[str, Callable[..., Any]] | None = None,
    executable_roles: tuple[str, ...] | list[str] | None = None,
    profile_skill_fallbacks: Mapping[
        str, Mapping[str, Iterable[str]]
    ]
    | None = None,
    profile_skill_provider_modes: Mapping[str, Mapping[str, str]] | None = None,
    skill_mode: str = "legacy",
    layer_settings: Mapping[str, Mapping[str, Any]] | None = None,
) -> SkillRegistry:
    registry = SkillRegistry(
        executable_roles=executable_roles,
        profile_skill_fallbacks=profile_skill_fallbacks,
        profile_skill_provider_modes=profile_skill_provider_modes,
        skill_mode=skill_mode,
        layer_settings=layer_settings,
    )
    for definition in discover_skills(
        project_root,
        runtime_overrides=runtime_overrides,
        lazy=skill_mode == "layered",
    ):
        registry.register(definition)
    return registry
