"""Small OpenAI-compatible client with explicit usage metadata."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass
class LLMToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    reasoning_chars: int = 0
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class LLMJSONError(ValueError):
    """JSON decoding failure that preserves provider usage metadata."""

    def __init__(self, message: str, response: LLMResponse):
        super().__init__(message)
        self.response = response


class ResearchLLM(Protocol):
    model_id: str

    async def complete_text(self, system: str, user: str, *, max_tokens: int) -> LLMResponse:
        ...

    async def complete_json(self, system: str, user: str, *, max_tokens: int) -> tuple[dict[str, Any], LLMResponse]:
        ...

    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        ...


class OpenAIResearchLLM:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        provider_id: str = "default",
        request_timeout_seconds: float = 180.0,
        max_retries: int = 0,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.model_id = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.provider_id = provider_id
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=request_timeout_seconds,
            max_retries=max_retries,
        )

    async def complete_text(
        self, system: str, user: str, *, max_tokens: int
    ) -> LLMResponse:
        return await self._complete(
            system,
            user,
            max_tokens=max_tokens,
        )

    async def _complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        json_mode: bool = False,
    ) -> LLMResponse:
        request_options: dict[str, Any] = {}
        if json_mode:
            request_options["response_format"] = {"type": "json_object"}
            # DeepSeek reasoning can consume the full completion allowance before
            # emitting JSON. Extraction needs a small, deterministic response.
            if "deepseek" in self.model_id.lower():
                request_options["extra_body"] = {"thinking": {"type": "disabled"}}

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=getattr(self, "temperature", 0.1),
            max_tokens=max_tokens,
            **request_options,
        )
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None) or ""
        usage = response.usage
        return LLMResponse(
            content=content,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason,
            reasoning_chars=len(reasoning_content),
        )

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = await self._complete(
            system + "\nReturn exactly one valid JSON object and no Markdown.",
            user,
            max_tokens=max_tokens,
            json_mode=True,
        )
        text = response.content.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as original_error:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                detail = "LLM did not return a JSON object"
                if not text and response.reasoning_chars:
                    detail += " (reasoning was emitted but final content was empty)"
                if response.finish_reason == "length":
                    detail += " (completion token limit reached)"
                raise LLMJSONError(detail, response) from original_error
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMJSONError("LLM returned malformed JSON", response) from exc
        if not isinstance(value, dict):
            raise LLMJSONError("LLM JSON response must be an object", response)
        return value, response

    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        request_options: dict[str, Any] = {}
        if "deepseek" in self.model_id.lower():
            request_options["extra_body"] = {"thinking": {"type": "disabled"}}
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=getattr(self, "temperature", 0.1),
            max_tokens=max_tokens,
            **request_options,
        )
        choice = response.choices[0]
        message = choice.message
        calls: list[LLMToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(arguments, dict):
                calls.append(
                    LLMToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )
        usage = response.usage
        reasoning_content = getattr(message, "reasoning_content", None) or ""
        return LLMResponse(
            content=message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=choice.finish_reason,
            reasoning_chars=len(reasoning_content),
            tool_calls=calls,
        )
