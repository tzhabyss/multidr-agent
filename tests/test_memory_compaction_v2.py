from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.memory import LayeredMemory
from deepresearch.memory_system.compaction import (
    CompactionController,
    apply_segment_to_rolling_state,
    extract_must_keep_spans,
    normalize_segment,
    project_memories,
)
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import ConversationMessage, RollingMemoryState


def _message(message_id: str, role: str, content: str) -> ConversationMessage:
    return ConversationMessage(
        message_id=message_id,
        turn_id=message_id,
        role=role,
        content=content,
    )


def test_default_memory_window_is_100k_with_60k_durable_compaction_boundary():
    config = load_runtime_config().memory

    assert config.context_window_tokens == 100_000
    assert config.summary_trigger_tokens == 60_000
    assert config.compaction_light_ratio == 0.60
    assert config.compaction_emergency_ratio == 0.95


def test_100k_window_uses_graduated_compaction_pressure():
    controller = CompactionController(
        context_window_tokens=100_000,
        light_ratio=0.60,
        query_ratio=0.75,
        summary_ratio=0.88,
        emergency_ratio=0.95,
    )

    assert controller.assess(59_999).level == "normal"
    assert controller.assess(60_000).level == "light"
    assert controller.assess(75_000).level == "query_aware"
    assert controller.assess(88_000).level == "summarize"
    assert controller.assess(95_000).level == "emergency"
    assert controller.assess(95_000).segment_multiplier == 4


def test_must_keep_uses_exact_user_spans_and_rejects_secrets():
    messages = [
        _message(
            "m1",
            "user",
            "不是只用BM25，改成BM25和Dense混合检索。测试预算必须控制在20万Token以内。",
        ),
        _message("m2", "assistant", "决定把数据库改成MySQL。"),
        _message(
            "m3",
            "user",
            "必须保存 API key = sk-abcdefghijklmnopqrstuv。",
        ),
    ]

    spans, rejected = extract_must_keep_spans(messages)

    assert rejected == 1
    assert {item.kind for item in spans} == {"correction", "constraint"}
    assert all(item.message_id == "m1" for item in spans)
    assert all(
        messages[0].content[item.start : item.end] == item.text for item in spans
    )
    assert not any("MySQL" in item.text for item in spans)
    assert not any("sk-" in item.text for item in spans)


def test_structured_segment_updates_rolling_state_without_summary_of_summary():
    messages = [
        _message(
            "m1",
            "user",
            "不对，不是只使用BM25，改成BM25和Dense混合检索。",
        )
    ]
    segment = normalize_segment(
        value={
            "abstract": "用户将检索方案改为混合检索。",
            "segment_summary": "用户纠正旧方案，决定采用BM25与Dense混合检索。",
            "overview": {
                "active_entities": ["BM25", "Dense"],
                "constraints": [],
                "decisions": ["采用BM25和Dense混合检索"],
                "corrections": ["旧的纯BM25方案已失效"],
                "unresolved": [],
            },
            "state_delta": {
                "add_decisions": ["采用BM25和Dense混合检索"],
                "add_corrections": ["旧的纯BM25方案已失效"],
                "supersede_decisions": [
                    {"old": "只使用BM25", "new": "采用BM25和Dense混合检索"}
                ],
            },
        },
        messages=messages,
        segment_id="segment-1",
        session_id="session-1",
        model_id="test-model",
    )
    state = RollingMemoryState(decisions=["只使用BM25"])

    updated = apply_segment_to_rolling_state(state, segment, segment_limit=12)

    assert updated.revision == 1
    assert updated.segment_ids == ["segment-1"]
    assert "只使用BM25" not in updated.decisions
    assert updated.decisions == ["采用BM25和Dense混合检索"]
    assert segment.must_keep_spans[0].text == messages[0].content


def test_structured_segment_redacts_secret_echo_without_losing_compaction():
    messages = [_message("m1", "user", "请不要记录我刚才粘贴的凭据。")]

    segment = normalize_segment(
        value={
            "abstract": "用户提供了凭据。",
            "segment_summary": "API key = sk-abcdefghijklmnopqrstuv",
        },
        messages=messages,
        segment_id="segment-secret",
        session_id="session-secret",
        model_id="test-model",
    )

    assert "sk-" not in segment.model_dump_json()
    assert "不得记录" in segment.summary
    assert segment.redacted_sensitive_fields >= 1


def test_exact_spans_merge_into_nonempty_state_delta():
    messages = [_message("m1", "user", "P95延迟上限必须是800毫秒。")]
    segment = normalize_segment(
        value={
            "abstract": "用户设置延迟上限。",
            "segment_summary": "P95 latency is capped at 800 ms.",
            "overview": {"constraints": ["P95 latency capped at 800 ms"]},
            "state_delta": {"add_constraints": ["P95 latency capped at 800 ms"]},
        },
        messages=messages,
        segment_id="segment-exact",
        session_id="session-exact",
        model_id="test-model",
    )
    updated = apply_segment_to_rolling_state(
        RollingMemoryState(), segment, segment_limit=12
    )

    assert "P95延迟上限必须是800毫秒。" in segment.state_delta.add_constraints
    assert "800毫秒" in "\n".join(updated.constraints)


def test_constraint_retraction_and_unresolved_priority_are_deterministic():
    messages = [_message("m1", "user", "Embedding模型还没确定，需要确认中英文召回效果。")]
    spans, _ = extract_must_keep_spans(messages)
    assert [item.kind for item in spans] == ["unresolved"]

    retract = normalize_segment(
        value={
            "abstract": "用户撤回博客禁令。",
            "segment_summary": "用户撤回禁止博客的约束。",
            "overview": {"corrections": ["撤回禁止博客"]},
            "state_delta": {
                "add_corrections": ["撤回禁止博客"],
                "retract_constraints": ["禁止引用博客"],
            },
        },
        messages=[_message("m2", "user", "撤回禁止引用博客这一条。")],
        segment_id="segment-retract",
        session_id="session-retract",
        model_id="test-model",
    )
    state = RollingMemoryState(
        constraints=["研究必须只使用官方来源", "禁止引用博客"]
    )
    updated = apply_segment_to_rolling_state(state, retract, segment_limit=12)
    assert updated.constraints == ["研究必须只使用官方来源"]


def test_resolve_all_unresolved_clears_prior_open_items():
    segment = normalize_segment(
        value={
            "abstract": "模型选择已经解决。",
            "segment_summary": "用户最终选择BGE-M3。",
            "overview": {"decisions": ["使用BGE-M3"]},
            "state_delta": {
                "add_decisions": ["使用BGE-M3"],
                "resolve_all_unresolved": True,
            },
        },
        messages=[_message("m1", "user", "已经确认，最终决定使用BGE-M3，模型问题已解决。")],
        segment_id="segment-resolved",
        session_id="session-resolved",
        model_id="test-model",
    )
    updated = apply_segment_to_rolling_state(
        RollingMemoryState(unresolved=["Embedding模型还没确定，需要验证中英文召回"]),
        segment,
        segment_limit=12,
    )
    assert updated.unresolved == []


def test_runtime_projection_is_query_aware_bounded_and_source_linked():
    result = project_memories(
        [
            {
                "memory_id": "segment-retrieval",
                "memory_type": "conversation_segment",
                "summary": "项目采用BM25和Dense混合检索。股票报告使用表格输出。",
                "retrieval_score": 0.95,
                "must_keep_spans": [
                    {
                        "kind": "constraint",
                        "text": "记忆检索必须使用一手测试数据",
                        "message_id": "m1",
                        "start": 0,
                        "end": 14,
                    }
                ],
            },
            {
                "memory_id": "episode-email",
                "memory_type": "research_episode",
                "summary": "之前把论文PDF发送到邮箱。",
                "retrieval_score": 0.20,
            },
        ],
        query="BM25 Dense记忆检索",
        role="planner",
        budget_tokens=200,
        max_items=1,
        sentence_limit=3,
    )

    assert result.source_memory_ids == ["segment-retrieval"]
    assert len(result.items) == 1
    assert "BM25" in result.items[0]["summary"]
    assert "一手测试数据" in result.items[0]["summary"]
    assert result.estimated_tokens <= 220


class StructuredCompactionLLM:
    model_id = "test-structured-compaction"

    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        assert "compress chat history" in system
        return (
            {
                "abstract": "用户纠正检索方案。",
                "segment_summary": "用户将方案从纯BM25纠正为BM25和Dense混合检索。",
                "overview": {
                    "active_entities": ["BM25", "Dense"],
                    "constraints": [],
                    "decisions": ["采用BM25和Dense混合检索"],
                    "corrections": ["纯BM25方案已失效"],
                    "unresolved": [],
                },
                "state_delta": {
                    "add_decisions": ["采用BM25和Dense混合检索"],
                    "add_corrections": ["纯BM25方案已失效"],
                },
            },
            LLMResponse("{}", input_tokens=100, output_tokens=50),
        )


@pytest.mark.asyncio
async def test_layered_memory_persists_v2_segment_before_pruning(tmp_path: Path):
    config = load_runtime_config().memory.model_copy(
        update={
            "local_db_path": str(tmp_path / "memory.sqlite"),
            "automatic_capture_enabled": False,
            "embedding_provider": "hash",
            "embedding_dimensions": 128,
            "recent_turns": 2,
            "segment_turns": 2,
            "summary_trigger_tokens": 1000,
        }
    )
    memory = LayeredMemory(
        config=config,
        root=tmp_path,
        llm=StructuredCompactionLLM(),
    )
    await memory.open()
    try:
        session = await memory.get_session("user-a", "session-a")
        contents = [
            "不对，不是只使用BM25，改成BM25和Dense混合检索。" + "x" * 600,
            "收到。" + "x" * 600,
            "继续讨论检索评测。" + "x" * 600,
            "好的。" + "x" * 600,
            "下一步做消融。" + "x" * 600,
        ]
        for index, content in enumerate(contents):
            session = await memory.append_message(
                session,
                role="user" if index % 2 == 0 else "assistant",
                content=content,
                turn_id=f"turn-{index}",
            )
        segments = await memory.facts.search_documents(
            "BM25 Dense",
            scope=memory.scope("user-a"),
            document_types={"conversation_segment"},
        )
    finally:
        await memory.close()

    assert len(session.messages) == 4
    assert session.rolling_state.revision == 1
    assert session.rolling_state.decisions == ["采用BM25和Dense混合检索"]
    metadata = segments[0].document.metadata
    assert metadata["schema_version"] == 2
    assert metadata["abstract"] == "用户纠正检索方案。"
    assert metadata["must_keep_spans"][0]["message_id"]
    assert metadata["detail_ref"]["message_ids"] == metadata["message_ids"]


@pytest.mark.asyncio
async def test_planner_packet_reinjects_active_preferences_and_constraints(tmp_path: Path):
    config = load_runtime_config().memory.model_copy(
        update={
            "local_db_path": str(tmp_path / "memory.sqlite"),
            "automatic_capture_enabled": False,
            "embedding_provider": "hash",
            "embedding_dimensions": 128,
        }
    )
    memory = LayeredMemory(
        config=config,
        root=tmp_path,
        llm=StructuredCompactionLLM(),
    )
    await memory.open()
    try:
        await memory.create_fact(
            user_id="user-a",
            content="用户偏好使用中文回答",
            category="preference",
        )
        await memory.create_fact(
            user_id="user-a",
            content="最终报告必须使用表格",
            category="constraint",
        )
        session = await memory.get_session("user-a", "session-new")
        packet = await memory.planner_packet(session, "研究量子芯片产业链")
    finally:
        await memory.close()

    assert "最终报告必须使用表格" in packet.user_constraints
    assert "用户偏好使用中文回答" in packet.preferences["remembered_preferences"]
    assert len(packet.source_memory_ids) == 2
