"""Bounded observe-act loop for role-scoped project skills."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .llm import LLMResponse, ResearchLLM
from .progress import emit_progress
from .skill_registry import SkillRegistry


MAX_OBSERVATION_ITEMS = 8
MAX_OBSERVATION_STRING_CHARS = 800


@dataclass
class SkillExecution:
    iteration: int
    skill_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class AgentLoopResult:
    executions: list[SkillExecution] = field(default_factory=list)
    responses: list[LLMResponse] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "model_finished"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, list):
        items = [_jsonable(item) for item in value[:MAX_OBSERVATION_ITEMS]]
        if len(value) > MAX_OBSERVATION_ITEMS:
            items.append({"_omitted_items": len(value) - MAX_OBSERVATION_ITEMS})
        return items
    if isinstance(value, tuple):
        return _jsonable(list(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str) and len(value) > MAX_OBSERVATION_STRING_CHARS:
        return value[:MAX_OBSERVATION_STRING_CHARS] + "...[truncated]"
    return value


class AgentLoop:
    """Run one agent's private skill loop without sharing mutable counters."""

    def __init__(
        self,
        *,
        llm: ResearchLLM,
        registry: SkillRegistry,
        role: str,
        profile_id: str,
        max_iterations: int,
        max_tool_calls: int,
        max_tool_calls_per_iteration: int | None = None,
        min_observation_rounds: int = 0,
        tool_deadline_at: float | None = None,
        progress_fields: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.role = role
        self.profile_id = profile_id
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_tool_calls_per_iteration = max_tool_calls_per_iteration
        self.min_observation_rounds = min_observation_rounds
        self.tool_deadline_at = tool_deadline_at
        self.progress_fields = progress_fields or {}

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tags: tuple[str, ...] = (),
        any_tags: tuple[str, ...] = (),
        max_tokens: int,
    ) -> AgentLoopResult:
        result = AgentLoopResult()
        operation_id = str(
            self.progress_fields.get("operation_id")
            or f"agent-loop:{self.role}"
        )
        event_fields = {
            key: value
            for key, value in self.progress_fields.items()
            if key != "operation_id"
        }
        emit_progress(
            "agent_loop_started",
            phase="loop",
            agent_role=self.role,
            status="started",
            message=(
                f"Agent Loop 开始：最多 {self.max_iterations} 轮、"
                f"{self.max_tool_calls} 次工具调用"
            ),
            operation_id=operation_id,
            metrics={
                "max_iterations": self.max_iterations,
                "max_tool_calls": self.max_tool_calls,
            },
            **event_fields,
        )
        complete_with_tools = getattr(self.llm, "complete_with_tools", None)
        if complete_with_tools is None or not getattr(
            self.llm, "supports_tool_calling", True
        ):
            result.stop_reason = "tool_calling_unavailable"
            emit_progress(
                "agent_loop_completed",
                phase="loop",
                agent_role=self.role,
                status="completed",
                message="Agent Loop 未执行：模型不支持工具调用",
                operation_id=operation_id,
                metrics={"iterations": 0, "tool_calls": 0, "stop_reason": result.stop_reason},
                **event_fields,
            )
            return result

        tools = self.registry.to_openai_tools(
            role=self.role,
            profile_id=self.profile_id,
            tags=tags,
            any_tags=any_tags,
        )
        if not tools:
            result.stop_reason = "no_executable_skills"
            emit_progress(
                "agent_loop_completed",
                phase="loop",
                agent_role=self.role,
                status="completed",
                message="Agent Loop 未执行：没有可调用的 Skill",
                operation_id=operation_id,
                metrics={"iterations": 0, "tool_calls": 0, "stop_reason": result.stop_reason},
                **event_fields,
            )
            return result
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        call_count = 0
        observation_followups_remaining = 0
        for iteration in range(1, self.max_iterations + 1):
            if self.tool_deadline_at is not None and time.time() >= self.tool_deadline_at:
                result.stop_reason = "stage_deadline_reached"
                break
            emit_progress(
                "agent_loop_iteration_started",
                phase="loop",
                agent_role=self.role,
                status="progress",
                message=f"Agent Loop 第 {iteration}/{self.max_iterations} 轮：等待模型决定下一步",
                operation_id=operation_id,
                metrics={
                    "iteration": iteration,
                    "max_iterations": self.max_iterations,
                    "tool_calls": call_count,
                },
                **event_fields,
            )
            try:
                response_call = complete_with_tools(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
                if self.tool_deadline_at is None:
                    response = await response_call
                else:
                    remaining_seconds = self.tool_deadline_at - time.time()
                    if remaining_seconds <= 0:
                        raise asyncio.TimeoutError("Agent-loop deadline reached")
                    response = await asyncio.wait_for(
                        response_call, timeout=remaining_seconds
                    )
            except asyncio.TimeoutError:
                result.stop_reason = "stage_deadline_reached"
                break
            except Exception:
                if self.profile_id not in {"general_research", "drb_evaluation"}:
                    raise
                # High-recall profiles retain the deterministic query fallback
                # in the worker. A provider rejection stops this optional loop
                # instead of discarding the entire research case.
                result.stop_reason = "model_error"
                break
            result.responses.append(response)
            tool_calls = list(getattr(response, "tool_calls", []) or [])
            trace: dict[str, Any] = {
                "iteration": iteration,
                "content": response.content,
                "finish_reason": response.finish_reason,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "tool_calls": [],
                "observations": [],
            }
            if not tool_calls:
                result.traces.append(trace)
                result.stop_reason = "model_finished"
                break

            remaining = self.max_tool_calls - call_count
            if remaining <= 0:
                result.traces.append(trace)
                result.stop_reason = "tool_budget_exhausted"
                break
            if self.tool_deadline_at is not None and time.time() >= self.tool_deadline_at:
                result.traces.append(trace)
                result.stop_reason = "stage_deadline_reached"
                break
            iteration_limit = self.max_tool_calls_per_iteration or remaining
            selected = tool_calls[: min(remaining, iteration_limit)]
            for call in selected:
                emit_progress(
                    "skill_started",
                    phase="loop",
                    agent_role=self.role,
                    status="progress",
                    message=f"Loop {iteration}/{self.max_iterations} 调用 {call.name}",
                    visibility="normal",
                    operation_id=operation_id,
                    metrics={
                        "iteration": iteration,
                        "skill_name": call.name,
                        "arguments": call.arguments,
                    },
                    **event_fields,
                )
            assistant_calls = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in selected
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": assistant_calls,
                }
            )

            async def execute(call: Any) -> SkillExecution:
                try:
                    value = await self.registry.execute(
                        role=self.role,
                        profile_id=self.profile_id,
                        skill_name=call.name,
                        arguments=call.arguments,
                    )
                    return SkillExecution(
                        iteration=iteration,
                        skill_name=call.name,
                        arguments=call.arguments,
                        result=value,
                    )
                except Exception as exc:
                    return SkillExecution(
                        iteration=iteration,
                        skill_name=call.name,
                        arguments=call.arguments,
                        error=f"{type(exc).__name__}: {exc}",
                    )

            execution_call = asyncio.gather(*(execute(call) for call in selected))
            try:
                if self.tool_deadline_at is None:
                    executions = await execution_call
                else:
                    remaining_seconds = self.tool_deadline_at - time.time()
                    if remaining_seconds <= 0:
                        raise asyncio.TimeoutError("Agent-loop deadline reached")
                    executions = await asyncio.wait_for(
                        execution_call, timeout=remaining_seconds
                    )
            except asyncio.TimeoutError:
                result.traces.append(trace)
                result.stop_reason = "stage_deadline_reached"
                break
            call_count += len(executions)
            for call, execution in zip(selected, executions):
                result.executions.append(execution)
                observation = (
                    {"success": False, "error": execution.error}
                    if execution.error
                    else {"success": True, "result": _jsonable(execution.result)}
                )
                trace["tool_calls"].append(
                    {"id": call.call_id, "name": call.name, "arguments": call.arguments}
                )
                trace["observations"].append(
                    {"name": call.name, **observation}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": json.dumps(observation, ensure_ascii=False, default=str),
                    }
                )
                emit_progress(
                    "skill_completed" if not execution.error else "skill_failed",
                    phase="loop",
                    agent_role=self.role,
                    status="progress",
                    message=(
                        f"{call.name} 调用完成"
                        if not execution.error
                        else f"{call.name} 调用失败：{execution.error}"
                    ),
                    visibility="verbose",
                    operation_id=operation_id,
                    metrics={
                        "iteration": iteration,
                        "skill_name": call.name,
                        "success": not bool(execution.error),
                    },
                    **event_fields,
                )
            result.traces.append(trace)
            if call_count >= self.max_tool_calls:
                observation_followups_remaining = max(
                    observation_followups_remaining,
                    self.min_observation_rounds,
                )
                if observation_followups_remaining and iteration < self.max_iterations:
                    observation_followups_remaining -= 1
                    continue
                result.stop_reason = "tool_budget_exhausted"
                break
        else:
            result.stop_reason = "max_iterations"
        emit_progress(
            "agent_loop_completed",
            phase="loop",
            agent_role=self.role,
            status="completed",
            message=(
                f"Agent Loop 结束：{result.stop_reason}，"
                f"共 {len(result.traces)} 轮、{call_count} 次工具调用"
            ),
            operation_id=operation_id,
            metrics={
                "iterations": len(result.traces),
                "tool_calls": call_count,
                "stop_reason": result.stop_reason,
                "input_tokens": sum(item.input_tokens for item in result.responses),
                "output_tokens": sum(item.output_tokens for item in result.responses),
            },
            **event_fields,
        )
        return result
