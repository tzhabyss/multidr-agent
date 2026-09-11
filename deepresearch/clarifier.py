"""Memory-aware preflight clarification before a research run is created."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .llm import ResearchLLM
from .prompts import CLARIFIER_SYSTEM_PROMPT, SESSION_RESPONSE_SYSTEM_PROMPT
from .runtime_config import RuntimeConfig
from .progress import emit_progress
from .schemas import ClarifierDecision, MemoryPacket, ResearchBrief, SessionMemory


class Clarifier:
    def __init__(self, llm: ResearchLLM, runtime_config: RuntimeConfig):
        self.llm = llm
        self.runtime_config = runtime_config

    async def clarify(
        self,
        *,
        message: str,
        session: SessionMemory,
        memory: MemoryPacket,
    ) -> ClarifierDecision:
        operation_id = f"clarifier:{session.session_id}"
        emit_progress(
            "clarifier_started",
            phase="clarifier",
            agent_role="clarifier",
            status="started",
            message="正在理解问题、会话上下文和用户约束",
            operation_id=operation_id,
        )
        payload = {
            "message": message,
            "pending_clarification": session.pending_clarification,
            "session_memory": memory.model_dump(mode="json"),
            "output_schema": {
                "action": "clarify | research | respond_from_context",
                "research_brief": {
                    "question": "self-contained resolved question",
                    "objective": "what the research must accomplish",
                    "assumptions": ["explicit assumption"],
                    "resolved_references": ["his -> named entity"],
                    "user_constraints": ["constraint"],
                },
                "question": "one clarification question or null",
                "response": "optional context-only answer or null",
                "active_entities": ["entity"],
                "user_corrections": ["explicit user correction"],
                "preferences": {"stable_preference": "value"},
                "memory_refs": ["episode id used only for context"],
            },
        }
        fallback_used = False
        response = None
        clarifier_started_at = time.monotonic()
        try:
            clarifier_policy = self.runtime_config.llm_recovery.policy_for("clarifier")
            clarifier_call = self.llm.complete_json(
                    CLARIFIER_SYSTEM_PROMPT,
                    json.dumps(payload, ensure_ascii=False),
                    max_tokens=self.runtime_config.llm_output_tokens.clarifier,
                )
            clarifier_timeout = (
                clarifier_policy.outer_timeout_seconds()
                if clarifier_policy is not None
                else None
            )
            value, response = (
                await asyncio.wait_for(clarifier_call, timeout=clarifier_timeout)
                if clarifier_timeout is not None
                else await clarifier_call
            )
            decision = ClarifierDecision.model_validate(value)
        except Exception:
            fallback_used = True
            decision = ClarifierDecision(
                action="research",
                research_brief=ResearchBrief(
                    question=message,
                    objective=message,
                    assumptions=[],
                    user_constraints=session.user_corrections,
                ),
                active_entities=session.active_entities,
            )

        if decision.action == "clarify" and not decision.question:
            decision = decision.model_copy(
                update={"question": "请补充你希望研究的具体对象或范围。"}
            )
        if decision.action == "research" and decision.research_brief is None:
            decision = decision.model_copy(
                update={
                    "research_brief": ResearchBrief(
                        question=message,
                        objective=message,
                        user_constraints=session.user_corrections,
                    )
                }
            )
        emit_progress(
            "clarifier_completed",
            phase="clarifier",
            agent_role="clarifier",
            status="completed",
            message=f"问题澄清完成，决定：{decision.action}",
            operation_id=operation_id,
            metrics={
                "action": decision.action,
                "fallback_used": fallback_used,
                "input_tokens": response.input_tokens if response is not None else 0,
                "output_tokens": response.output_tokens if response is not None else 0,
                "duration_seconds": round(
                    time.monotonic() - clarifier_started_at, 3
                ),
            },
        )
        return decision

    async def respond_from_context(
        self, *, message: str, session: SessionMemory, memory: MemoryPacket
    ) -> str:
        operation_id = f"context-response:{session.session_id}"
        emit_progress(
            "context_response_started",
            phase="memory",
            agent_role="clarifier",
            status="started",
            message="正在根据当前会话上下文生成回答",
            operation_id=operation_id,
        )
        if not memory.recent_dialogue and not memory.session_summary:
            emit_progress(
                "context_response_completed",
                phase="memory",
                agent_role="clarifier",
                status="completed",
                message="当前会话上下文不足",
                operation_id=operation_id,
            )
            return "当前会话里没有足够信息，需要发起一次研究。"
        payload: dict[str, Any] = {
            "message": message,
            "session_summary": memory.session_summary,
            "recent_dialogue": memory.recent_dialogue,
            "user_constraints": memory.user_constraints,
        }
        try:
            response = await self.llm.complete_text(
                SESSION_RESPONSE_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=self.runtime_config.llm_output_tokens.session_response,
            )
            emit_progress(
                "context_response_completed",
                phase="memory",
                agent_role="clarifier",
                status="completed",
                message="会话上下文回答已生成",
                operation_id=operation_id,
            )
            return response.content.strip() or "当前会话里没有足够信息，需要发起一次研究。"
        except Exception:
            emit_progress(
                "context_response_failed",
                phase="memory",
                agent_role="clarifier",
                status="failed",
                message="会话上下文回答生成失败",
                operation_id=operation_id,
            )
            return "当前会话里没有足够信息，需要发起一次研究。"
