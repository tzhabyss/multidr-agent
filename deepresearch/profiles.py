"""Profile manifests and fail-closed tool authorization."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from .runtime_config import AgentProfileRuntimeConfig, RuntimeConfig


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class Profile:
    profile_id: str
    enabled_packs: frozenset[str]
    allowed_tools: frozenset[str]
    default_dimensions: tuple[str, ...]
    policy_set: str = "public_read_only"
    runtime_policy: AgentProfileRuntimeConfig = field(
        default_factory=AgentProfileRuntimeConfig
    )


GENERAL_PROFILE = Profile(
    profile_id="general_research",
    enabled_packs=frozenset({"common_web"}),
    allowed_tools=frozenset({"web_search", "safe_fetch"}),
    default_dimensions=("核心事实与背景", "主要观点与证据", "限制、冲突与风险"),
    runtime_policy=AgentProfileRuntimeConfig(
        max_tool_calls_per_iteration=4,
        min_observation_rounds=1,
        allow_derived_analysis=True,
        skill_evolution_mode="observe",
        skill_fallbacks={
            "web-search": ["deepseek-web-search", "anysearch-search"]
        },
        skill_provider_modes={"web-search": "fanout"},
        mode_overrides={
            "standard": {
                "max_workers": 8,
                "max_total_searches": 200,
                "max_total_fetches": 336,
                "max_wall_time_seconds": 3600,
                "research_soft_wall_time_seconds": 3300,
                "writer_wall_time_seconds": 1220,
                "reviewer_wall_time_seconds": 1020,
                "writer_rewrite_wall_time_seconds": 1220,
                "reviewer_recheck_wall_time_seconds": 500,
                "artifact_reserve_seconds": 60,
                "max_total_wall_time_seconds": 7620,
            }
        },
    ),
)


DRB_EVALUATION_PROFILE = Profile(
    profile_id="drb_evaluation",
    enabled_packs=frozenset({"common_web"}),
    allowed_tools=frozenset({"web_search", "safe_fetch"}),
    default_dimensions=("题目要求与事实数据", "比较、推导与机制分析", "指定结构与呈现格式"),
    runtime_policy=AgentProfileRuntimeConfig(
        semantic_citation_gate="advisory",
        requirement_matrix=True,
        exact_url_blocklist=True,
        enforce_candidate_quality_floor=True,
        max_tool_calls_per_iteration=4,
        min_observation_rounds=1,
        preserve_rich_markdown=True,
        include_audit_links=False,
        allow_derived_analysis=True,
        skill_evolution_mode="off",
        skill_fallbacks={
            "web-search": ["deepseek-web-search", "anysearch-search"]
        },
        skill_provider_modes={"web-search": "fanout"},
        mode_overrides={
            "standard": {
                "max_workers": 8,
                "max_total_searches": 200,
                "max_total_fetches": 336,
                "max_wall_time_seconds": 3600,
                "research_soft_wall_time_seconds": 3300,
                "writer_wall_time_seconds": 1220,
                "reviewer_wall_time_seconds": 1020,
                "writer_rewrite_wall_time_seconds": 1220,
                "reviewer_recheck_wall_time_seconds": 500,
                "artifact_reserve_seconds": 60,
                "max_total_wall_time_seconds": 7620,
            }
        },
    ),
)


class ProfileRegistry:
    def __init__(
        self,
        profiles: list[Profile] | None = None,
        *,
        runtime_config: RuntimeConfig | None = None,
    ):
        selected = profiles or [GENERAL_PROFILE, DRB_EVALUATION_PROFILE]
        if runtime_config is not None:
            selected = [
                Profile(
                    profile_id=item.profile_id,
                    enabled_packs=item.enabled_packs,
                    allowed_tools=item.allowed_tools,
                    default_dimensions=item.default_dimensions,
                    policy_set=item.policy_set,
                    runtime_policy=runtime_config.profile_config(item.profile_id),
                )
                for item in selected
                if item.profile_id in runtime_config.agent_profiles
            ]
        self._profiles = {profile.profile_id: profile for profile in selected}

    def get(self, profile_id: str) -> Profile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise AuthorizationError(f"Unknown or disabled profile: {profile_id}") from exc


@dataclass
class ToolSpec:
    name: str
    pack: str
    function: Callable[..., Any]
    input_model: type[BaseModel] | None = None
    timeout_seconds: float = 30.0
    read_only: bool = True


class AuthorizedToolRegistry:
    def __init__(self, profiles: ProfileRegistry):
        self.profiles = profiles
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.read_only:
            raise AuthorizationError("V1 only permits read-only tools")
        self._tools[spec.name] = spec

    async def execute(
        self,
        *,
        profile_id: str,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        profile = self.profiles.get(profile_id)
        spec = self._tools.get(tool_name)
        if spec is None:
            raise AuthorizationError(f"Unknown tool: {tool_name}")
        if spec.pack not in profile.enabled_packs:
            raise AuthorizationError(
                f"Tool pack {spec.pack!r} is disabled for profile {profile_id!r}"
            )
        if tool_name not in profile.allowed_tools:
            raise AuthorizationError(
                f"Tool {tool_name!r} is not allowed for profile {profile_id!r}"
            )
        if agent_id != "worker":
            raise AuthorizationError(f"Agent {agent_id!r} may not execute research tools")

        kwargs = arguments
        if spec.input_model is not None:
            kwargs = spec.input_model.model_validate(arguments).model_dump()

        async def call() -> Any:
            if inspect.iscoroutinefunction(spec.function):
                return await spec.function(**kwargs)
            return await asyncio.to_thread(spec.function, **kwargs)

        return await asyncio.wait_for(call(), timeout=spec.timeout_seconds)
