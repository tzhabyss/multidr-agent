"""Role-scoped skill registry with fail-closed execution."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping
from jsonschema import validators, ValidationError
from contextvars import ContextVar
from .tool_runtime import current_runtime, operation_context


_CAPABILITY_GRANULARITIES = {"strategic", "functional", "atomic"}


class SkillError(RuntimeError):
    pass


class SkillArgumentError(SkillError, ValueError):
    pass


class SkillAuthorizationError(PermissionError):
    pass


@dataclass
class SkillDefinition:
    skill_id: str
    description: str
    instructions: str
    roles: frozenset[str]
    profiles: frozenset[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None
    fallback_skills: tuple[str, ...] = ()
    timeout_seconds: float = 30.0
    tags: frozenset[str] = field(default_factory=frozenset)
    source_path: str = ""
    enabled: bool = True
    result_kind: str = "search_hits"
    skill_layer: str = "strategy"
    # Runtime layer answers how the Skill is loaded. Capability granularity
    # answers at what semantic scope it may evolve; the two dimensions must
    # remain independent.
    capability_granularity: str = "functional"
    skill_version: str = "1.0.0"
    applicable_entity_types: frozenset[str] = field(default_factory=frozenset)
    applicable_source_families: frozenset[str] = field(default_factory=frozenset)
    required_tools: frozenset[str] = field(default_factory=frozenset)
    output_artifact_roles: frozenset[str] = field(default_factory=frozenset)
    failure_signatures: frozenset[str] = field(default_factory=frozenset)
    cost_class: str = "medium"
    instructions_path: str = ""
    instructions_loader: Callable[[], str] | None = field(
        default=None, repr=False, compare=False
    )
    handler_loader: Callable[[], Callable[..., Any] | None] | None = field(
        default=None, repr=False, compare=False
    )
    instructions_loaded: bool = False
    handler_loaded: bool = False

    def __post_init__(self) -> None:
        if self.capability_granularity not in _CAPABILITY_GRANULARITIES:
            raise SkillError(
                "Unknown capability granularity: "
                f"{self.capability_granularity}"
            )
        if not self.skill_version.strip():
            raise SkillError("skill_version must not be empty")

    @property
    def tool_name(self) -> str:
        return self.skill_id.replace("-", "_")

    @property
    def executable(self) -> bool:
        return (
            self.handler is not None
            or self.handler_loader is not None
            or bool(self.fallback_skills)
        )

    @property
    def evolution_candidate_policy(self) -> str:
        if self.capability_granularity == "atomic":
            return "automatic_candidate_allowed"
        if self.capability_granularity == "functional":
            return "episode_requires_review"
        return "observe_only"

    def load_instructions(self) -> str:
        """Load and cache the SKILL.md body only when a prompt selects it."""

        if not self.instructions_loaded and self.instructions_loader is not None:
            self.instructions = self.instructions_loader().strip()
            self.instructions_loaded = True
        return self.instructions

    def load_handler(self) -> Callable[..., Any] | None:
        """Import and cache an executable entrypoint on its first invocation."""

        if not self.handler_loaded and self.handler_loader is not None:
            self.handler = self.handler_loader()
            self.handler_loaded = True
        return self.handler


def _matches_identity(definition: SkillDefinition, value: str) -> bool:
    return value in {definition.skill_id, definition.tool_name}


class SkillRegistry:
    """Stores declarative skills and enforces role/profile boundaries."""

    @property
    def _skills(self):
        frozen = self._snapshot_skills.get()
        return self._active_skills if frozen is None else frozen

    @_skills.setter
    def _skills(self, value):
        self._active_skills = value

    @property
    def _tools(self):
        return {item.tool_name: item for item in self._skills.values()}

    def __init__(
        self,
        *,
        executable_roles: Iterable[str] | None = None,
        profile_skill_fallbacks: Mapping[
            str, Mapping[str, Iterable[str]]
        ]
        | None = None,
        profile_skill_provider_modes: Mapping[str, Mapping[str, str]] | None = None,
        skill_mode: str = "legacy",
        layer_settings: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._snapshot_skills = ContextVar(f"skill_registry_{id(self)}", default=None)
        self._skills: dict[str, SkillDefinition] = {}
        self._executable_roles = frozenset(executable_roles or {"worker"})
        self._profile_skill_fallbacks = {
            profile_id: {
                skill_name: tuple(fallback_names)
                for skill_name, fallback_names in skill_fallbacks.items()
            }
            for profile_id, skill_fallbacks in (
                profile_skill_fallbacks or {}
            ).items()
        }
        self._profile_skill_provider_modes = {
            profile_id: dict(modes)
            for profile_id, modes in (profile_skill_provider_modes or {}).items()
        }
        if skill_mode not in {"legacy", "layered"}:
            raise SkillError(f"Unknown skill mode: {skill_mode}")
        self.skill_mode = skill_mode
        self._layer_settings = {
            str(layer): dict(settings)
            for layer, settings in (layer_settings or {}).items()
        }

    def register(self, definition: SkillDefinition) -> None:
        self._validate_arguments_schema(definition.parameters)
        if definition.skill_id in self._skills:
            raise SkillError(f"Duplicate skill id: {definition.skill_id}")
        if any(item.tool_name == definition.tool_name for item in self._skills.values()):
            raise SkillError(f"Duplicate skill tool name: {definition.tool_name}")
        self._skills[definition.skill_id] = definition

    def get(self, name: str) -> SkillDefinition | None:
        return next(
            (item for item in self._skills.values() if _matches_identity(item, name)),
            None,
        )

    def _profile_allows_definition(
        self, definition: SkillDefinition, profile_id: str
    ) -> bool:
        """Apply a profile's provider chain to direct child-tool visibility too."""

        for parent_name, allowed_children in self._profile_skill_fallbacks.get(
            profile_id, {}
        ).items():
            parent = self.get(parent_name)
            if parent is None or definition.skill_id not in parent.fallback_skills:
                continue
            return definition.skill_id in allowed_children
        return True

    def list(
        self,
        *,
        role: str,
        profile_id: str,
        executable_only: bool = False,
        tags: Iterable[str] = (),
        any_tags: Iterable[str] = (),
        layers: Iterable[str] = (),
        entity_type: str = "",
        source_family: str = "",
        output_artifact_role: str = "",
    ) -> list[SkillDefinition]:
        required_tags = set(tags)
        alternative_tags = set(any_tags)
        allowed_layers = set(layers)
        return [
            item
            for item in self._skills.values()
            if item.enabled
            and self._layer_enabled(item.skill_layer)
            and self._layer_allows_skill(item.skill_layer, item.skill_id)
            and role in item.roles
            and (not item.profiles or profile_id in item.profiles)
            and self._profile_allows_definition(item, profile_id)
            and (
                not executable_only
                or (item.executable and role in self._executable_roles)
            )
            and required_tags.issubset(item.tags)
            and (not alternative_tags or bool(alternative_tags & item.tags))
            and (not allowed_layers or item.skill_layer in allowed_layers)
            and (
                not entity_type
                or not item.applicable_entity_types
                or entity_type in item.applicable_entity_types
            )
            and (
                not source_family
                or not item.applicable_source_families
                or source_family in item.applicable_source_families
            )
            and (
                not output_artifact_role
                or not item.output_artifact_roles
                or output_artifact_role in item.output_artifact_roles
            )
        ]

    def _layer_enabled(self, layer: str) -> bool:
        return bool(self._layer_settings.get(layer, {}).get("enabled", True))

    def _layer_allows_skill(self, layer: str, skill_id: str) -> bool:
        configured = self._layer_settings.get(layer, {}).get("skills")
        if not configured:
            return True
        return skill_id in {str(item) for item in configured}

    def metadata_catalog(
        self,
        *,
        role: str,
        profile_id: str,
        layers: Iterable[str] = (),
        exclude_skills: Iterable[str] = (),
    ) -> str:
        """Return a compact discovery catalog without loading any Skill body."""

        excluded = set(exclude_skills)
        rows = [
            f"- {item.skill_id}: {item.description}"
            for item in self.list(
                role=role,
                profile_id=profile_id,
                layers=layers,
            )
            if item.skill_id not in excluded
        ]
        if not rows:
            return ""
        return "## Available strategy skills (metadata only)\n" + "\n".join(rows)

    def summary_for(
        self,
        *,
        role: str,
        profile_id: str,
        selected_skills: Iterable[str],
    ) -> str:
        """Return selected Skill identities/descriptions without repeating bodies."""

        selected = set(selected_skills)
        rows = [
            f"- {item.skill_id}: {item.description}"
            for item in self.list(role=role, profile_id=profile_id)
            if item.skill_id in selected
        ]
        return "" if not rows else "## Selected strategy skills\n" + "\n".join(rows)

    def instructions_for(
        self,
        *,
        role: str,
        profile_id: str,
        layers: Iterable[str] = (),
        selected_skills: Iterable[str] | None = None,
    ) -> str:
        selected = None if selected_skills is None else set(selected_skills)
        items = self.list(role=role, profile_id=profile_id, layers=layers)
        if selected is not None:
            items = [item for item in items if item.skill_id in selected]
        sections: list[str] = []
        for item in items:
            body = item.load_instructions().strip()
            if body:
                sections.append(
                    f"## Skill: {item.skill_id} [{item.skill_layer}]\n{body}"
                )
        return "\n\n".join(sections)

    def prompt_instructions(
        self,
        *,
        role: str,
        profile_id: str,
        phase: str = "default",
        selected_skills: Iterable[str] = (),
    ) -> str:
        """Project the minimum Skill context required by one responsibility."""

        if self.skill_mode == "legacy":
            return self.instructions_for(role=role, profile_id=profile_id)
        selected = tuple(dict.fromkeys(str(item) for item in selected_skills if item))
        if phase == "worker_strategy":
            return self.metadata_catalog(
                role=role,
                profile_id=profile_id,
                layers=("strategy",),
                exclude_skills=("evidence-extraction",),
            )
        if phase == "worker_loop":
            return self.instructions_for(
                role=role,
                profile_id=profile_id,
                layers=("strategy",),
                selected_skills=selected,
            )
        if phase == "worker_reflection":
            return self.summary_for(
                role=role,
                profile_id=profile_id,
                selected_skills=selected,
            )
        if phase == "extraction":
            return self.instructions_for(
                role=role,
                profile_id=profile_id,
                layers=("strategy",),
                selected_skills=("evidence-extraction",),
            )
        # Planner/Synthesizer/Writer/Reviewer receive only their core prompt by
        # default.  The orchestrator layer remains an explicit opt-in extension.
        layer = self._layer_settings.get("orchestrator", {})
        if phase in {"planner", "synthesizer"} and layer.get("injection") == "playbook":
            return self.instructions_for(
                role=role,
                profile_id=profile_id,
                layers=("orchestrator",),
                selected_skills=layer.get("skills", ()),
            )
        return ""

    def loading_snapshot(self) -> list[dict[str, Any]]:
        """Expose load state for tests and run-level audit diagnostics."""

        return [
            {
                "skill_id": item.skill_id,
                "skill_layer": item.skill_layer,
                "capability_granularity": item.capability_granularity,
                "skill_version": item.skill_version,
                "evolution_candidate_policy": item.evolution_candidate_policy,
                "instructions_loaded": item.instructions_loaded,
                "handler_loaded": item.handler_loaded,
            }
            for item in self._skills.values()
        ]

    def provider_call_width(self, *, profile_id: str, skill_name: str) -> int:
        """Return provider invocations consumed by one logical skill call.

        The value is deterministic from runtime policy. Fanout calls every
        configured child, while fallback and direct tools invoke at most one
        provider at a time.
        """

        definition = self.get(skill_name)
        if definition is None:
            return 1
        fallback_skills = self._profile_skill_fallbacks.get(profile_id, {}).get(
            definition.skill_id, definition.fallback_skills
        )
        provider_mode = self._profile_skill_provider_modes.get(profile_id, {}).get(
            definition.skill_id, "fallback"
        )
        if definition.handler is None and fallback_skills and provider_mode == "fanout":
            return max(1, len(fallback_skills))
        return 1

    def to_openai_tools(
        self,
        *,
        role: str,
        profile_id: str,
        tags: Iterable[str] = (),
        any_tags: Iterable[str] = (),
        layers: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        effective_layers = tuple(layers)
        if self.skill_mode == "layered" and not effective_layers:
            effective_layers = ("access",)
        return [
            {
                "type": "function",
                "function": {
                    "name": item.tool_name,
                    "description": item.description,
                    "parameters": item.parameters
                    or {"type": "object", "properties": {}, "additionalProperties": False},
                },
            }
            for item in self.list(
                role=role,
                profile_id=profile_id,
                executable_only=True,
                tags=tags,
                any_tags=any_tags,
                layers=effective_layers,
            )
        ]

    async def execute(self, *, role: str, profile_id: str, skill_name: str,
                      arguments: dict[str, Any]) -> Any:
        runtime = current_runtime()
        definition = self.get(skill_name)
        async def call():
            return await self._execute(role=role, profile_id=profile_id,
                                       skill_name=skill_name, arguments=arguments)
        if runtime is None:
            return await call()
        def preflight():
            if definition is None or definition not in self.list(
                role=role, profile_id=profile_id, executable_only=True
            ):
                raise SkillAuthorizationError(f"Unavailable skill: {skill_name}")
            self._validate_arguments(definition, arguments)
        with operation_context(profile_id=profile_id, role=role):
            return await runtime.execute(
                definition.skill_id if definition else skill_name, call, arguments=arguments,
                version=definition.skill_version if definition else "",
                identity=definition.source_path if definition else "",
                kind="skill" if definition and definition.fallback_skills and not definition.handler else "tool",
                source=definition.source_path if definition else "", preflight=preflight,
            )

    async def _execute(
        self,
        *,
        role: str,
        profile_id: str,
        skill_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        definition = self.get(skill_name)
        if definition is None or not definition.enabled:
            raise SkillAuthorizationError(f"Unknown or disabled skill: {skill_name}")
        if role not in definition.roles:
            raise SkillAuthorizationError(
                f"Role {role!r} may not execute skill {definition.skill_id!r}"
            )
        if role not in self._executable_roles:
            raise SkillAuthorizationError(
                f"Role {role!r} is not permitted to execute tools by runtime policy"
            )
        if definition.profiles and profile_id not in definition.profiles:
            raise SkillAuthorizationError(
                f"Profile {profile_id!r} may not execute skill {definition.skill_id!r}"
            )
        if not self._profile_allows_definition(definition, profile_id):
            raise SkillAuthorizationError(
                f"Profile {profile_id!r} does not select skill "
                f"{definition.skill_id!r} in its provider chain"
            )
        self._validate_arguments(definition, arguments)
        fallback_skills = self._profile_skill_fallbacks.get(profile_id, {}).get(
            definition.skill_id, definition.fallback_skills
        )
        handler = definition.load_handler()
        if handler is None and fallback_skills:
            provider_mode = self._profile_skill_provider_modes.get(profile_id, {}).get(
                definition.skill_id, "fallback"
            )
            if provider_mode == "fanout":

                async def execute_child(
                    child_name: str,
                ) -> tuple[str, Any, Exception | None]:
                    child = self.get(child_name)
                    if child is None or child not in self.list(role=role, profile_id=profile_id, executable_only=True):
                        return child_name, None, SkillError("provider unavailable")
                    try:
                        value = await self.execute(
                            role=role,
                            profile_id=profile_id,
                            skill_name=child.skill_id,
                            arguments=arguments,
                        )
                        return child.skill_id, value, None
                    except Exception as exc:
                        if isinstance(exc, (PermissionError, SkillAuthorizationError)):
                            raise
                        return child.skill_id, None, exc

                tasks = [asyncio.create_task(execute_child(child_name)) for child_name in fallback_skills]
                try:
                    outcomes = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                merged: list[Any] = []
                seen: set[str] = set()
                errors: list[str] = []
                for child_name, value, error in outcomes:
                    if error is not None:
                        errors.append(f"{child_name}: {type(error).__name__}: {error}")
                        continue
                    rows = value if isinstance(value, list) else [value]
                    for row in rows:
                        if row is None:
                            continue
                        if isinstance(row, dict):
                            identity = str(row.get("url") or row)
                        else:
                            identity = str(getattr(row, "url", "") or row)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        merged.append(row)
                if merged:
                    return merged
                raise SkillError("All fanout skills failed: " + "; ".join(errors))
            errors: list[str] = []
            for child_name in fallback_skills:
                child = self.get(child_name)
                if child is None or child not in self.list(role=role, profile_id=profile_id, executable_only=True):
                    continue
                try:
                    result = await self.execute(
                        role=role,
                        profile_id=profile_id,
                        skill_name=child.skill_id,
                        arguments=arguments,
                    )
                    if result:
                        return result
                    errors.append(f"{child.skill_id}: empty")
                except Exception as exc:
                    if isinstance(exc, (PermissionError, SkillAuthorizationError)):
                        raise
                    errors.append(f"{child.skill_id}: {type(exc).__name__}: {exc}")
            raise SkillError("All fallback skills failed: " + "; ".join(errors))
        handler = definition.load_handler()
        if handler is None:
            raise SkillError(f"Skill {definition.skill_id!r} is instruction-only")

        async def call() -> Any:
            if inspect.iscoroutinefunction(handler):
                return await handler(**arguments)
            return await asyncio.to_thread(handler, **arguments)

        return await asyncio.wait_for(call(), timeout=definition.timeout_seconds)

    @staticmethod
    def _validate_arguments_schema(schema):
        # Remote tool schemas may only refer to definitions inside this schema.
        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"$ref", "$dynamicRef"} and (not isinstance(item, str) or not item.startswith("#")):
                        raise SkillError("External schema references are not permitted")
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(schema)
        validators.validator_for(schema).check_schema(schema)

    @staticmethod
    def _validate_arguments(
        definition: SkillDefinition, arguments: dict[str, Any]
    ) -> None:
        schema = definition.parameters or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = sorted(required - set(arguments))
        if missing:
            raise SkillArgumentError(
                f"Skill {definition.skill_id!r} missing arguments: {', '.join(missing)}"
            )
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                raise SkillArgumentError(
                    f"Skill {definition.skill_id!r} received unknown arguments: "
                    + ", ".join(unknown)
                )
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for name, value in arguments.items():
            expected_name = properties.get(name, {}).get("type")
            expected = type_map.get(expected_name) if isinstance(expected_name, str) else None
            if expected and (isinstance(value, bool) and expected_name in {"integer", "number"}):
                raise SkillArgumentError(f"Skill argument {name!r} must be {expected_name}")
            if expected and not isinstance(value, expected):
                raise SkillArgumentError(f"Skill argument {name!r} must be {expected_name}")
            rule = properties.get(name, {})
            if isinstance(value, int):
                if "minimum" in rule and value < rule["minimum"]:
                    raise SkillArgumentError(f"Skill argument {name!r} is below its minimum")
                if "maximum" in rule and value > rule["maximum"]:
                    raise SkillArgumentError(f"Skill argument {name!r} exceeds its maximum")
            if "enum" in rule and value not in rule["enum"]:
                raise SkillArgumentError(f"Skill argument {name!r} is not an allowed value")
        try:
            validators.validator_for(schema)(schema).validate(arguments)
        except ValidationError as exc:
            raise SkillArgumentError(f"Skill {definition.skill_id!r} invalid arguments at {list(exc.path)}: {exc.validator}") from exc
