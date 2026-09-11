from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.llm import LLMResponse
from deepresearch.memory import LayeredMemory
from deepresearch.runtime_config import load_runtime_config
from deepresearch.service import ResearchService
from tests.fakes import FakeFetcher, FakeLLM, FakeSearchBackend


class SessionAwareFakeLLM(FakeLLM):
    async def complete_json(self, system: str, user: str, *, max_tokens: int = 2000):
        if "conversation clarifier" in system:
            payload = json.loads(user)
            message = payload["message"]
            if "shorter" in message.lower():
                return (
                    {
                        "action": "respond_from_context",
                        "response": "The verified fact is 42 units.",
                        "active_entities": ["verified fact"],
                    },
                    LLMResponse("{}", input_tokens=10, output_tokens=5),
                )
            return (
                {
                    "action": "research",
                    "research_brief": {
                        "question": "What is the verified fact?",
                        "objective": "Find the verified fact",
                        "assumptions": [],
                        "resolved_references": [],
                        "user_constraints": [],
                    },
                    "active_entities": ["verified fact"],
                    "user_corrections": ["Do not guess"],
                    "preferences": {"answer_style": "concise"},
                },
                LLMResponse("{}", input_tokens=10, output_tokens=5),
            )
        return await super().complete_json(system, user, max_tokens=max_tokens)


@pytest.mark.asyncio
async def test_one_session_supports_multiple_turns_and_task_scoped_memory(tmp_path: Path):
    config = load_runtime_config()
    config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={
                    "local_db_path": str(tmp_path / "memory.sqlite"),
                    "automatic_capture_enabled": False,
                }
            )
        }
    )
    search = FakeSearchBackend()
    async with ResearchService.in_memory(
        llm=SessionAwareFakeLLM(),
        search_backend=search,
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    ) as service:
        first = await service.chat("What is the verified fact?")
        second = await service.chat("Make that answer shorter.", session_id=first.session_id)
        assert service.memory is not None
        session = await service.memory.get_session("local-user", first.session_id)
        archived = await service.memory.local.archived_messages(
            "local-user", config.memory.project_id, first.session_id
        )
        new_session = await service.memory.get_session("local-user", "session-later")
        planner_memory = await service.memory.planner_packet(
            new_session, "verified fact"
        )

    assert first.action == "research"
    assert first.run is not None
    assert second.action == "respond_from_context"
    assert second.run is None
    assert second.session_id == first.session_id
    assert len(session.messages) == 4
    assert session.messages[1].content != first.message
    assert first.run.run_id in session.messages[1].content
    assert archived[1]["content"] == first.message
    assert session.related_run_ids == [first.run.run_id]
    assert len(search.calls) == 1
    assert planner_memory.relevant_episodes
    assert planner_memory.relevant_episodes[0]["run_id"] == first.run.run_id
    assert "Do not guess" in planner_memory.user_constraints
    assert planner_memory.preferences["answer_style"] == "concise"

    run_dir = Path(first.run.artifact_dir)
    assert sorted(path.name for path in run_dir.iterdir()) == ["details", "report.md"]
    assert (run_dir / "details" / "traces" / "extraction.json").exists()
    assert (run_dir / "details" / "audit" / "audit-log.json").exists()


@pytest.mark.asyncio
async def test_sessions_are_isolated_by_user_and_session(tmp_path: Path):
    config = load_runtime_config()
    config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={
                    "local_db_path": str(tmp_path / "memory.sqlite"),
                    "automatic_capture_enabled": False,
                }
            )
        }
    )
    async with ResearchService.in_memory(
        llm=SessionAwareFakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    ) as service:
        a = await service.memory.get_session("user-a", "session-shared")
        b = await service.memory.get_session("user-b", "session-shared")
        a = await service.memory.append_message(
            a, role="user", content="private-a", turn_id="turn-a"
        )
        b = await service.memory.get_session("user-b", "session-shared")

    assert [item.content for item in a.messages] == ["private-a"]
    assert b.messages == []


@pytest.mark.asyncio
async def test_failed_compaction_never_discards_unsummarized_messages(tmp_path: Path):
    config = load_runtime_config().memory.model_copy(
        update={
            "local_db_path": str(tmp_path / "memory.sqlite"),
            "automatic_capture_enabled": False,
            "recent_turns": 2,
            "summary_trigger_tokens": 1000,
        }
    )
    memory = LayeredMemory(
        config=config,
        root=tmp_path,
        llm=SessionAwareFakeLLM(),
    )
    await memory.open()
    try:
        session = await memory.get_session("user-a", "session-compaction")
        for index in range(5):
            session = await memory.append_message(
                session,
                role="user" if index % 2 == 0 else "assistant",
                content=f"message-{index}-" + "x" * 500,
                turn_id=f"turn-{index}",
            )
    finally:
        await memory.close()

    assert len(session.messages) == 5
    assert session.rolling_summary == ""
