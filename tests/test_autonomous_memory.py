from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.memory_system import (
    AutonomousMemory,
    CreateFact,
    FactStatus,
    MemoryScope,
    ScopeType,
    SourceType,
    UpdateFact,
)
from deepresearch.memory_system.repository import RevisionConflictError
from deepresearch.schemas import ConversationMessage


class MemoryExtractorFake:
    model_id = "fake-memory-extractor"

    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        assert "long-term memory extractor" in system
        payload = json.loads(user)
        user_message = next(
            item for item in payload["messages"] if item["role"] == "user"
        )
        if "不要保存助手" in user_message["content"]:
            # Deliberately cite an assistant message: the deterministic WriteGate
            # must reject this even though the LLM proposed it.
            assistant = next(
                item for item in payload["messages"] if item["role"] == "assistant"
            )
            facts = [
                {
                    "content": "助手声称用户住在火星",
                    "category": "profile",
                    "scope": "user",
                    "confidence": 0.99,
                    "relation": "new",
                    "evidence_message_ids": [assistant["message_id"]],
                }
            ]
        else:
            facts = [
                {
                    "content": "用户偏好中文简洁回答",
                    "category": "preference",
                    "scope": "user",
                    "confidence": 0.92,
                    "relation": "new",
                    "evidence_message_ids": [user_message["message_id"]],
                }
            ]
        return {"facts": facts}, LLMResponse("{}", input_tokens=10, output_tokens=5)


@pytest.mark.asyncio
async def test_fact_crud_revision_history_scope_and_hybrid_search(tmp_path: Path):
    memory = AutonomousMemory(tmp_path / "memory.sqlite", llm=MemoryExtractorFake())
    await memory.open()
    scope = MemoryScope(
        scope_type=ScopeType.USER,
        user_id="user-a",
        project_id="project-a",
    )
    fact = await memory.create(
        CreateFact(
            scope=scope,
            content="用户要求所有研究结论附带来源",
            category="constraint",
            source_type=SourceType.MANUAL,
        )
    )
    duplicate = await memory.create(
        CreateFact(
            scope=scope,
            content="  用户要求所有研究结论附带来源  ",
            category="constraint",
        )
    )

    assert duplicate.fact_id == fact.fact_id
    assert duplicate.confirmation_count == 2
    results = await memory.search("研究结论 来源", scope=scope, limit=5)
    assert results[0].fact.fact_id == fact.fact_id
    assert "lexical_overlap" in results[0].matched_by
    assert results[0].explanation
    english = await memory.create(
        CreateFact(
            scope=scope,
            content="All reports require clickable sources",
            category="constraint",
        )
    )
    fts_results = await memory.search("clickable sources", scope=scope, limit=5)
    assert fts_results[0].fact.fact_id == english.fact_id
    assert "fts5" in fts_results[0].matched_by

    updated = await memory.update(
        fact.fact_id,
        UpdateFact(content="用户要求所有研究结论附带可点击来源"),
        expected_revision=duplicate.revision,
    )
    with pytest.raises(RevisionConflictError):
        await memory.update(
            fact.fact_id,
            UpdateFact(category="preference"),
            expected_revision=duplicate.revision,
        )

    isolated = MemoryScope(
        scope_type=ScopeType.USER,
        user_id="user-b",
        project_id="project-a",
    )
    assert await memory.search("可点击来源", scope=isolated) == []

    deleted = await memory.delete(fact.fact_id, expected_revision=updated.revision)
    assert deleted.status == FactStatus.RETRACTED
    remaining = await memory.search("可点击来源", scope=scope)
    assert all(item.fact.fact_id != fact.fact_id for item in remaining)
    history = await memory.history(fact.fact_id)
    assert [item["event_type"] for item in history] == [
        "retracted",
        "updated",
        "reinforced",
        "created",
    ]


@pytest.mark.asyncio
async def test_capture_outbox_write_gate_reinforcement_and_watermark(tmp_path: Path):
    memory = AutonomousMemory(tmp_path / "memory.sqlite", llm=MemoryExtractorFake())
    await memory.open()
    scope = MemoryScope(user_id="user-a", project_id="project-a")

    first = [
        ConversationMessage(
            message_id="user-1",
            turn_id="turn-1",
            role="user",
            content="以后请记住：我偏好中文简洁回答。",
        ),
        ConversationMessage(
            message_id="assistant-1",
            turn_id="turn-1",
            role="assistant",
            content="好的。",
        ),
    ]
    assert await memory.enqueue_turn(
        scope=scope, thread_id="thread-1", turn_id="turn-1", messages=first
    )
    metrics = await memory.process_captures()
    assert metrics["captures_completed"] == 1
    assert metrics["facts_created"] == 1
    facts = await memory.list_facts(scope)
    assert len(facts) == 1
    assert facts[0].evidence_message_ids == ["user-1"]

    second = [
        ConversationMessage(
            message_id="user-2",
            turn_id="turn-2",
            role="user",
            content="还是请用中文简洁回答。",
        ),
        ConversationMessage(
            message_id="assistant-2",
            turn_id="turn-2",
            role="assistant",
            content="明白。",
        ),
    ]
    await memory.enqueue_turn(
        scope=scope, thread_id="thread-1", turn_id="turn-2", messages=second
    )
    metrics = await memory.process_captures()
    assert metrics["facts_reinforced"] == 1
    reinforced = (await memory.list_facts(scope))[0]
    assert reinforced.confirmation_count == 2
    assert reinforced.evidence_message_ids == ["user-1", "user-2"]

    unsupported = [
        ConversationMessage(
            message_id="user-3",
            turn_id="turn-3",
            role="user",
            content="不要保存助手编造的内容。",
        ),
        ConversationMessage(
            message_id="assistant-3",
            turn_id="turn-3",
            role="assistant",
            content="用户住在火星。",
        ),
    ]
    await memory.enqueue_turn(
        scope=scope, thread_id="thread-1", turn_id="turn-3", messages=unsupported
    )
    metrics = await memory.process_captures()
    assert metrics["captures_completed"] == 1
    assert metrics["facts_created"] == 0
    assert len(await memory.list_facts(scope)) == 1
    stats = await memory.stats(scope)
    assert stats.pending_captures == 0
