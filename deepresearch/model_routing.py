"""Mode-aware routing from LLM responsibilities to configured providers."""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable, TypeVar

from .llm import LLMResponse, OpenAIResearchLLM, ResearchLLM
from .runtime_config import RuntimeConfig


T = TypeVar("T")
logger = logging.getLogger(__name__)


class ModelRouter:
    """Resolve stage proxies without coupling model quality to tool authority."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        injected_llm: ResearchLLM | None = None,
    ) -> None:
        self.config = config
        self.injected_llm = injected_llm
        self._mode: ContextVar[str] = ContextVar(
            "deepresearch_model_mode", default=config.default_mode
        )
        self._clients: dict[tuple[str, str, float], ResearchLLM] = {}

    @property
    def current_mode(self) -> str:
        return self.config.normalize_mode(self._mode.get())

    def activate(self, mode: str) -> Token[str]:
        return self._mode.set(self.config.normalize_mode(mode))

    def reset(self, token: Token[str]) -> None:
        self._mode.reset(token)

    def for_stage(self, stage: str) -> "StageRoutedLLM":
        # Validate eagerly so configuration errors fail during service startup.
        self.config.stage_tier(stage, self.config.default_mode)
        return StageRoutedLLM(self, stage)

    def for_tier(self, tier_name: str, *, role: str = "judge") -> "TierRoutedLLM":
        """Return a retrying tier proxy for roles outside the research graph."""

        if tier_name not in self.config.model_tiers:
            raise ValueError(f"Unknown model tier: {tier_name}")
        return TierRoutedLLM(self, tier_name, role)

    def resolve(self, stage: str, mode: str | None = None) -> ResearchLLM:
        if self.injected_llm is not None:
            return self.injected_llm
        tier_name = self.config.stage_tier(stage, mode or self.current_mode)
        return self.resolve_tier(tier_name)

    def resolve_tier(self, tier_name: str) -> ResearchLLM:
        """Resolve a logical tier for non-graph roles such as benchmark judging."""
        if self.injected_llm is not None:
            return self.injected_llm
        if tier_name not in self.config.model_tiers:
            raise ValueError(f"Unknown model tier: {tier_name}")
        tier = self.config.model_tiers[tier_name]
        provider = self.config.llm_providers[tier.provider]
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise ValueError(
                f"{provider.api_key_env} is required by LLM provider {tier.provider!r}"
            )
        base_url = os.getenv(provider.base_url_env) or provider.base_url
        model = (
            (os.getenv(tier.model_env) if tier.model_env else None)
            or tier.model
            or os.getenv("OPENAI_MODEL")
            or "gpt-4.1-mini"
        )
        key = (tier.provider, model, tier.temperature)
        if key not in self._clients:
            self._clients[key] = OpenAIResearchLLM(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=tier.temperature,
                provider_id=tier.provider,
                request_timeout_seconds=provider.request_timeout_seconds,
                max_retries=provider.max_retries,
            )
        return self._clients[key]

    def _candidate_tiers(self, primary_tier: str, role: str) -> list[str]:
        policy = self.config.llm_recovery.policy_for(role)
        if policy is None:
            return [primary_tier]
        return list(dict.fromkeys([primary_tier, *policy.fallback_tiers]))

    @staticmethod
    def _is_fast_transient_failure(exc: Exception) -> bool:
        """Identify failures that are worth retrying on the same deployment."""

        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is not None:
            try:
                status = int(status)
            except (TypeError, ValueError):
                status = None
        if status is not None:
            return status in {408, 409, 425, 429} or status >= 500
        name = type(exc).__name__.casefold()
        text = str(exc).casefold()
        if any(
            marker in name or marker in text
            for marker in (
                "connection",
                "connecterror",
                "timeout",
                "temporar",
                "rate limit",
                "server error",
                "service unavailable",
                "reset by peer",
                "remoteprotocol",
            )
        ):
            return True
        return isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError, OSError))

    @staticmethod
    async def _cancel_tasks(*tasks: asyncio.Task[Any] | None) -> None:
        existing = [task for task in tasks if task is not None]
        for task in existing:
            if not task.done():
                task.cancel()
        if existing:
            # Retrieve exceptions from already-completed losers as well as
            # waiting for cancellation cleanup from active HTTP requests.
            await asyncio.gather(*existing, return_exceptions=True)

    @staticmethod
    def _most_informative_error(errors: list[Exception]) -> Exception | None:
        """Prefer a completed provider response over a later timeout wrapper.

        JSON/model errors carry an ``LLMResponse`` with usage, finish reason,
        and raw content. Losing that error to a fallback timeout made completed
        Planner calls look like unexplained hangs and prevented artifact-level
        diagnosis.
        """

        for exc in errors:
            if isinstance(getattr(exc, "response", None), LLMResponse):
                return exc
        for exc in errors:
            if not isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                return exc
        return errors[0] if errors else None

    async def _call_hedged(
        self,
        *,
        primary_tier: str,
        role: str,
        invoke: Callable[[ResearchLLM], Awaitable[T]],
    ) -> T:
        policy = self.config.llm_recovery.policy_for(role)
        assert policy is not None
        tiers = self._candidate_tiers(primary_tier, role)
        if len(tiers) < 2 or policy.attempt_timeout_seconds is None:
            return await self._call_with_recovery_sequential(
                primary_tier=primary_tier,
                role=role,
                invoke=invoke,
            )

        loop = asyncio.get_running_loop()
        primary_timeout = float(policy.attempt_timeout_seconds)
        fallback_timeout = float(
            policy.fallback_attempt_timeout_seconds or primary_timeout
        )
        hedge_after = float(policy.hedge_after_seconds or primary_timeout)
        grace = float(policy.primary_result_grace_seconds)
        fallback_index = 0
        fast_retries_used = 0
        primary_task: asyncio.Task[T] | None = None
        fallback_task: asyncio.Task[T] | None = None
        errors: list[Exception] = []

        async def run_tier(tier_name: str, timeout: float) -> T:
            call = invoke(self.resolve_tier(tier_name))
            return await asyncio.wait_for(call, timeout=timeout)

        def start_primary() -> asyncio.Task[T]:
            return asyncio.create_task(run_tier(primary_tier, primary_timeout))

        def start_fallback() -> asyncio.Task[T] | None:
            nonlocal fallback_index
            if fallback_index >= len(tiers) - 1:
                return None
            fallback_index += 1
            return asyncio.create_task(
                run_tier(tiers[fallback_index], fallback_timeout)
            )

        primary_task = start_primary()
        primary_started = loop.time()
        try:
            while True:
                hedge_wait = max(0.0, hedge_after - (loop.time() - primary_started))
                hedge_task = asyncio.create_task(asyncio.sleep(hedge_wait))
                done, _ = await asyncio.wait(
                    {primary_task, hedge_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if primary_task in done:
                    await self._cancel_tasks(hedge_task)
                    try:
                        return primary_task.result()
                    except Exception as exc:
                        errors.append(exc)
                        elapsed = loop.time() - primary_started
                        if (
                            elapsed <= policy.fast_failure_window_seconds
                            and fast_retries_used < policy.fast_failure_retries
                            and self._is_fast_transient_failure(exc)
                        ):
                            fast_retries_used += 1
                            logger.warning(
                                "LLM stage %s tier %s failed quickly; retrying primary %d/%d: %s: %s",
                                role,
                                primary_tier,
                                fast_retries_used,
                                policy.fast_failure_retries,
                                type(exc).__name__,
                                exc,
                            )
                            primary_task = start_primary()
                            primary_started = loop.time()
                            continue
                        fallback_task = start_fallback()
                        if fallback_task is None:
                            raise
                        break
                else:
                    await self._cancel_tasks(hedge_task)
                    fallback_task = start_fallback()
                    if fallback_task is None:
                        break
                    logger.info(
                        "LLM stage %s starting fallback tier %s while primary tier %s continues",
                        role,
                        tiers[fallback_index],
                        primary_tier,
                    )
                    break

            # Race the outstanding primary request against fallback requests.
            while primary_task is not None or fallback_task is not None:
                active = {
                    task for task in (primary_task, fallback_task) if task is not None
                }
                if not active:
                    break
                done, _ = await asyncio.wait(
                    active,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if primary_task is not None and primary_task in done:
                    try:
                        value = primary_task.result()
                    except Exception as exc:
                        errors.append(exc)
                        primary_task = None
                        continue
                    await self._cancel_tasks(fallback_task)
                    return value

                if fallback_task is not None and fallback_task in done:
                    try:
                        fallback_value = fallback_task.result()
                    except Exception as exc:
                        errors.append(exc)
                        fallback_task = start_fallback()
                        if fallback_task is None and primary_task is None:
                            break
                        continue
                    if primary_task is not None and not primary_task.done() and grace > 0:
                        try:
                            primary_value = await asyncio.wait_for(
                                asyncio.shield(primary_task), timeout=grace
                            )
                        except Exception:
                            primary_value = None
                        if primary_value is not None:
                            await self._cancel_tasks(fallback_task)
                            return primary_value
                    await self._cancel_tasks(primary_task)
                    return fallback_value

                if primary_task is not None and primary_task.done():
                    primary_task = None
                if fallback_task is not None and fallback_task.done():
                    fallback_task = None
        finally:
            await self._cancel_tasks(primary_task, fallback_task)
        selected_error = self._most_informative_error(errors)
        if selected_error is not None:
            raise selected_error
        raise RuntimeError(f"LLM stage {role} hedged recovery produced no result")

    async def _call_with_recovery_sequential(
        self,
        *,
        primary_tier: str,
        role: str,
        invoke: Callable[[ResearchLLM], Awaitable[T]],
    ) -> T:
        """Sequential implementation retained for non-hedged stages."""

        policy = self.config.llm_recovery.policy_for(role)
        attempts_per_model = policy.max_attempts_per_model if policy else 1
        tiers = self._candidate_tiers(primary_tier, role)
        last_error: Exception | None = None
        failure_index = 0
        for tier_name in tiers:
            for attempt in range(1, attempts_per_model + 1):
                try:
                    call = invoke(self.resolve_tier(tier_name))
                    if policy and policy.attempt_timeout_seconds is not None:
                        return await asyncio.wait_for(
                            call,
                            timeout=policy.attempt_timeout_seconds,
                        )
                    return await call
                except Exception as exc:
                    last_error = exc
                    failure_index += 1
                    logger.warning(
                        "LLM stage %s tier %s attempt %d/%d failed: %s: %s",
                        role,
                        tier_name,
                        attempt,
                        attempts_per_model,
                        type(exc).__name__,
                        exc,
                    )
                    is_last = (
                        tier_name == tiers[-1]
                        and failure_index >= len(tiers) * attempts_per_model
                    )
                    if is_last:
                        raise
                    if policy and policy.backoff_initial_seconds:
                        delay = min(
                            policy.backoff_initial_seconds
                            * (policy.backoff_multiplier ** (failure_index - 1)),
                            policy.backoff_max_seconds,
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _call_with_recovery(
        self,
        *,
        primary_tier: str,
        role: str,
        invoke: Callable[[ResearchLLM], Awaitable[T]],
    ) -> T:
        # Tests and embedded callers intentionally inject one concrete fake.
        # Preserve their historical one-call behavior and exception identity.
        if self.injected_llm is not None:
            return await invoke(self.injected_llm)

        policy = self.config.llm_recovery.policy_for(role)
        if (
            policy
            and policy.recovery_strategy == "hedged"
            and primary_tier in policy.hedged_primary_tiers
        ):
            return await self._call_hedged(
                primary_tier=primary_tier,
                role=role,
                invoke=invoke,
            )
        if policy and policy.recovery_strategy == "hedged":
            # Long hedges are reserved for the configured high-quality tier.
            # Quick/lower-tier modes keep their own wall-time contract and do
            # not inherit a 900-second Planner wait from Standard mode.
            call = invoke(self.resolve_tier(primary_tier))
            timeout = policy.attempt_timeout_seconds
            if role in {
                "planner",
                "worker_agent_loop",
                "synthesizer",
                "report_writer",
            }:
                timeout = min(
                    float(
                        timeout
                        or self.config.mode_config(self.current_mode).max_wall_time_seconds
                    ),
                    float(
                        self.config.mode_config(self.current_mode).max_wall_time_seconds
                    ),
                )
            return await asyncio.wait_for(call, timeout=timeout)
        return await self._call_with_recovery_sequential(
            primary_tier=primary_tier,
            role=role,
            invoke=invoke,
        )

    async def call_stage(
        self,
        stage: str,
        invoke: Callable[[ResearchLLM], Awaitable[T]],
    ) -> T:
        primary_tier = self.config.stage_tier(stage, self.current_mode)
        from .tool_runtime import current_runtime, operation_context
        runtime = current_runtime()
        if runtime is not None:
            async def observed_invoke(target):
                return await runtime.execute(
                    f"{stage}:{target.model_id}", lambda: invoke(target), kind="llm",
                    provider_id=str(getattr(target, "provider_id", "injected")),
                )
            with operation_context(model_stage=stage):
                return await runtime.execute(stage, lambda: self._call_with_recovery(
                    primary_tier=primary_tier, role=stage, invoke=observed_invoke,
                ), kind="stage")
        return await self._call_with_recovery(
            primary_tier=primary_tier,
            role=stage,
            invoke=invoke,
        )

    async def call_tier(
        self,
        tier_name: str,
        role: str,
        invoke: Callable[[ResearchLLM], Awaitable[T]],
    ) -> T:
        return await self._call_with_recovery(
            primary_tier=tier_name,
            role=role,
            invoke=invoke,
        )

    def describe(self, mode: str | None = None) -> dict[str, dict[str, str]]:
        selected_mode = self.config.normalize_mode(mode or self.current_mode)
        stages = self.config.mode_config(selected_mode).stages
        result: dict[str, dict[str, str]] = {}
        for stage in type(stages).model_fields:
            tier_name = str(getattr(stages, stage))
            tier = self.config.model_tiers[tier_name]
            model = (
                (os.getenv(tier.model_env) if tier.model_env else None)
                or tier.model
                or os.getenv("OPENAI_MODEL")
                or "gpt-4.1-mini"
            )
            result[stage] = {
                "tier": tier_name,
                "provider": tier.provider,
                "model": model,
            }
        return result


class StageRoutedLLM:
    """ResearchLLM proxy that resolves its target at call time."""

    def __init__(self, router: ModelRouter, stage: str) -> None:
        self.router = router
        self.stage = stage

    @property
    def model_id(self) -> str:
        return self.router.resolve(self.stage).model_id

    @property
    def provider_id(self) -> str:
        return str(getattr(self.router.resolve(self.stage), "provider_id", "injected"))

    @property
    def supports_tool_calling(self) -> bool:
        return callable(getattr(self.router.resolve(self.stage), "complete_with_tools", None))

    async def complete_text(
        self, system: str, user: str, *, max_tokens: int
    ) -> LLMResponse:
        return await self.router.call_stage(
            self.stage,
            lambda target: target.complete_text(system, user, max_tokens=max_tokens),
        )

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int
    ) -> tuple[dict[str, Any], LLMResponse]:
        return await self.router.call_stage(
            self.stage,
            lambda target: target.complete_json(system, user, max_tokens=max_tokens),
        )

    async def complete_json_validated(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        validator: Callable[[dict[str, Any]], None],
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Race only outputs that pass the calling stage's coarse schema check."""

        async def invoke(
            target: ResearchLLM,
        ) -> tuple[dict[str, Any], LLMResponse]:
            value, response = await target.complete_json(
                system, user, max_tokens=max_tokens
            )
            validator(value)
            return value, response

        return await self.router.call_stage(self.stage, invoke)

    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        async def invoke(target: ResearchLLM) -> LLMResponse:
            complete_with_tools = getattr(target, "complete_with_tools", None)
            if not callable(complete_with_tools):
                raise AttributeError("Underlying LLM does not support tool calling")
            return await complete_with_tools(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            )

        return await self.router.call_stage(
            self.stage,
            invoke,
        )


class TierRoutedLLM:
    """Retry/failover proxy rooted at an explicitly selected logical tier."""

    def __init__(self, router: ModelRouter, tier_name: str, role: str) -> None:
        self.router = router
        self.tier_name = tier_name
        self.role = role

    @property
    def model_id(self) -> str:
        return self.router.resolve_tier(self.tier_name).model_id

    @property
    def provider_id(self) -> str:
        return str(
            getattr(self.router.resolve_tier(self.tier_name), "provider_id", "injected")
        )

    async def complete_text(
        self, system: str, user: str, *, max_tokens: int
    ) -> LLMResponse:
        return await self.router.call_tier(
            self.tier_name,
            self.role,
            lambda target: target.complete_text(system, user, max_tokens=max_tokens),
        )

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int
    ) -> tuple[dict[str, Any], LLMResponse]:
        return await self.router.call_tier(
            self.tier_name,
            self.role,
            lambda target: target.complete_json(system, user, max_tokens=max_tokens),
        )

    async def complete_json_validated(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        validator: Callable[[dict[str, Any]], None],
    ) -> tuple[dict[str, Any], LLMResponse]:
        async def invoke(
            target: ResearchLLM,
        ) -> tuple[dict[str, Any], LLMResponse]:
            value, response = await target.complete_json(
                system, user, max_tokens=max_tokens
            )
            validator(value)
            return value, response

        return await self.router.call_tier(self.tier_name, self.role, invoke)

    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        async def invoke(target: ResearchLLM) -> LLMResponse:
            complete_with_tools = getattr(target, "complete_with_tools", None)
            if not callable(complete_with_tools):
                raise AttributeError("Underlying LLM does not support tool calling")
            return await complete_with_tools(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            )

        return await self.router.call_tier(self.tier_name, self.role, invoke)
