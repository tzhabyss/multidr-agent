from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Sequence

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.memory import LayeredMemory
from deepresearch.memory_system import (
    AutonomousMemory,
    CreateFact,
    MemoryDocument,
    MemoryScope,
)
from deepresearch.runtime_config import load_runtime_config


class TinySemanticProvider:
    model_id = "test:tiny-semantic-v1"
    dimensions = 3
    degraded = False

    def initialize(self) -> None:
        return None

    @staticmethod
    def _vector(value: str) -> list[float]:
        lowered = value.casefold()
        if "猫" in value or "cat" in lowered or "feline" in lowered:
            return [1.0, 0.0, 0.0]
        if "预算" in value or "budget" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def encode_query(self, value: str) -> list[float]:
        return self._vector(value)

    def encode_documents(self, values: Sequence[str]) -> list[list[float]]:
        return [self._vector(value) for value in values]


class CountingSemanticProvider(TinySemanticProvider):
    def __init__(self) -> None:
        self.query_calls = 0

    def encode_query(self, value: str) -> list[float]:
        self.query_calls += 1
        return super().encode_query(value)


class MinimalLLM:
    model_id = "test-minimal"

    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        if "compress chat history" in system:
            return (
                {
                    "summary": "长期摘要：用户要求保留预算约束。",
                    "segment_summary": "用户要求保留预算约束。",
                    "active_entities": ["预算"],
                    "user_corrections": [],
                    "decisions": ["保留预算约束"],
                    "unresolved": [],
                },
                LLMResponse("{}", input_tokens=5, output_tokens=5),
            )
        return {"facts": []}, LLMResponse("{}", input_tokens=5, output_tokens=5)


@pytest.mark.asyncio
async def test_true_provider_retrieves_cross_language_fact_and_document(tmp_path: Path):
    memory = AutonomousMemory(
        tmp_path / "semantic.sqlite",
        llm=MinimalLLM(),
        provider=TinySemanticProvider(),
        embedding_dimensions=3,
    )
    await memory.open()
    scope = MemoryScope(user_id="user-a", project_id="project-a")
    fact = await memory.create(
        CreateFact(scope=scope, content="用户喜欢猫科动物", category="preference")
    )
    await memory.upsert_document(
        MemoryDocument(
            document_id="episode-a",
            document_type="research_episode",
            scope=scope,
            content="预算需要逐项核对",
            metadata={"summary": "预算需要逐项核对"},
        )
    )

    fact_results = await memory.search("feline preference", scope=scope)
    document_results = await memory.search_documents(
        "check the budget", scope=scope, document_types={"research_episode"}
    )

    assert fact_results[0].fact.fact_id == fact.fact_id
    assert "semantic_vector" in fact_results[0].matched_by
    assert document_results[0].document.document_id == "episode-a"
    assert "semantic_vector" in document_results[0].matched_by
    with sqlite3.connect(tmp_path / "semantic.sqlite") as connection:
        assert connection.execute(
            "SELECT model FROM memory_embeddings WHERE fact_id = ?", (fact.fact_id,)
        ).fetchone()[0] == TinySemanticProvider.model_id


@pytest.mark.asyncio
async def test_prepared_query_is_encoded_once_across_retrieval_channels(
    tmp_path: Path,
):
    provider = CountingSemanticProvider()
    memory = AutonomousMemory(
        tmp_path / "shared-query.sqlite",
        llm=MinimalLLM(),
        provider=provider,
        embedding_dimensions=3,
    )
    await memory.open()
    scope = MemoryScope(user_id="user-a", project_id="project-a")
    fact = await memory.create(
        CreateFact(scope=scope, content="用户喜欢猫科动物", category="preference")
    )
    await memory.upsert_document(
        MemoryDocument(
            document_id="episode-budget",
            document_type="research_episode",
            scope=scope,
            content="猫科动物研究预算需要逐项核对",
        )
    )

    prepared = await memory.prepare_retrieval_query("feline budget")
    fact_results, document_results = await asyncio.gather(
        memory.search(
            "feline budget",
            scope=scope,
            prepared=prepared,
            record_usage=False,
        ),
        memory.search_documents(
            "feline budget",
            scope=scope,
            document_types={"research_episode"},
            prepared=prepared,
        ),
    )
    await memory.record_retrieval_usage(
        [item.fact.fact_id for item in fact_results]
    )

    assert provider.query_calls == 1
    assert fact_results[0].fact.fact_id == fact.fact_id
    assert document_results[0].document.document_id == "episode-budget"
    with sqlite3.connect(tmp_path / "shared-query.sqlite") as connection:
        assert connection.execute(
            "SELECT retrieval_count FROM memory_fact_usage WHERE fact_id = ?",
            (fact.fact_id,),
        ).fetchone()[0] == 1


@pytest.mark.asyncio
async def test_prepared_query_rejects_mismatched_text(tmp_path: Path):
    memory = AutonomousMemory(
        tmp_path / "mismatched-query.sqlite",
        llm=MinimalLLM(),
        provider=TinySemanticProvider(),
        embedding_dimensions=3,
    )
    await memory.open()
    prepared = await memory.prepare_retrieval_query("feline")

    with pytest.raises(ValueError, match="does not match"):
        await memory.search(
            "budget",
            scope=MemoryScope(user_id="user-a", project_id="project-a"),
            prepared=prepared,
        )


@pytest.mark.asyncio
async def test_layered_retrieval_reuses_one_query_embedding(tmp_path: Path):
    config = load_runtime_config().memory.model_copy(
        update={
            "local_db_path": str(tmp_path / "layered-shared-query.sqlite"),
            "automatic_capture_enabled": False,
            "embedding_provider": "hash",
            "embedding_dimensions": 3,
        }
    )
    memory = LayeredMemory(config=config, root=tmp_path, llm=MinimalLLM())
    provider = CountingSemanticProvider()
    memory.facts.embedding_provider = provider
    await memory.open()
    try:
        scope = memory.scope("user-a")
        fact = await memory.facts.create(
            CreateFact(
                scope=scope,
                content="用户喜欢猫科动物",
                category="preference",
            )
        )
        await memory.facts.upsert_document(
            MemoryDocument(
                document_id="episode-layered-budget",
                document_type="research_episode",
                scope=scope,
                content="猫科动物研究预算需要逐项核对",
            )
        )
        session = await memory.get_session("user-a", "session-a")

        results = await memory.relevant_episodes(
            session, "feline budget", limit=5
        )

        assert provider.query_calls == 1
        assert {item["memory_type"] for item in results} >= {
            "fact",
            "research_episode",
        }
        with sqlite3.connect(config.local_db_path) as connection:
            assert connection.execute(
                "SELECT retrieval_count FROM memory_fact_usage WHERE fact_id = ?",
                (fact.fact_id,),
            ).fetchone()[0] == 1
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_compaction_archives_raw_messages_and_indexes_immutable_segment(
    tmp_path: Path,
):
    config = load_runtime_config().memory.model_copy(
        update={
            "local_db_path": str(tmp_path / "segments.sqlite"),
            "automatic_capture_enabled": False,
            "embedding_provider": "hash",
            "embedding_dimensions": 128,
            "recent_turns": 2,
            "segment_turns": 2,
            "summary_trigger_tokens": 1000,
        }
    )
    memory = LayeredMemory(config=config, root=tmp_path, llm=MinimalLLM())
    await memory.open()
    try:
        session = await memory.get_session("user-a", "session-a")
        for index in range(5):
            content = f"message-{index}-" + "预算约束" + "x" * 500
            session = await memory.append_message(
                session,
                role="user" if index % 2 == 0 else "assistant",
                content=content,
                archive_content=content + "-raw",
                turn_id=f"turn-{index}",
            )
        archived = await memory.local.archived_messages(
            "user-a", config.project_id, "session-a"
        )
        segments = await memory.facts.search_documents(
            "预算约束",
            scope=memory.scope("user-a"),
            document_types={"conversation_segment"},
        )
    finally:
        await memory.close()

    assert len(session.messages) == 4
    assert len(archived) == 5
    assert archived[0]["content"].endswith("-raw")
    assert segments[0].document.metadata["message_ids"]
    assert segments[0].document.metadata["summary"] == "用户要求保留预算约束。"


def test_semantic_dedup_guard_rejects_numeric_and_negation_conflicts():
    assert AutonomousMemory._relation_guard_compatible(
        "预算上限是 100 元", "预算上限是 100 元"
    )
    assert not AutonomousMemory._relation_guard_compatible(
        "预算上限是 100 元", "预算上限是 200 元"
    )
    assert not AutonomousMemory._relation_guard_compatible(
        "允许保存报告", "不要保存报告"
    )
