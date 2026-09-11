from __future__ import annotations

from types import SimpleNamespace

import pytest

from deepresearch.llm import LLMJSONError, OpenAIResearchLLM


class FakeCompletions:
    def __init__(self, *, content: str | None, reasoning: str = "", finish_reason: str = "stop"):
        self.content = content
        self.reasoning = reasoning
        self.finish_reason = finish_reason
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content, reasoning_content=self.reasoning)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        usage = SimpleNamespace(prompt_tokens=41, completion_tokens=17)
        return SimpleNamespace(choices=[choice], usage=usage)


def make_llm(completions: FakeCompletions, model: str = "deepseek-v4-pro"):
    llm = OpenAIResearchLLM.__new__(OpenAIResearchLLM)
    llm.model_id = model
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm


@pytest.mark.asyncio
async def test_deepseek_json_completion_disables_reasoning_and_requests_json():
    completions = FakeCompletions(content='{"ok": true}')
    value, response = await make_llm(completions).complete_json("system", "user", max_tokens=300)

    assert value == {"ok": True}
    assert response.input_tokens == 41
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_json_error_preserves_empty_reasoning_response_usage():
    completions = FakeCompletions(
        content=None,
        reasoning="unfinished reasoning",
        finish_reason="length",
    )

    with pytest.raises(LLMJSONError) as caught:
        await make_llm(completions).complete_json("system", "user", max_tokens=300)

    assert caught.value.response.output_tokens == 17
    assert caught.value.response.reasoning_chars == len("unfinished reasoning")
    assert "final content was empty" in str(caught.value)
    assert "token limit reached" in str(caught.value)

