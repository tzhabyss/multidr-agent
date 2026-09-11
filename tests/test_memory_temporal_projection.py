from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.memory import LayeredMemory
from deepresearch.memory_system import MemoryDocument
from deepresearch.schemas import RollingMemoryState, SessionMemory
from deepresearch.runtime_config import load_runtime_config
from tests.fakes import FakeLLM


def _memory(tmp_path: Path) -> LayeredMemory:
    config = load_runtime_config().memory.model_copy(
        update={
            "local_db_path": str(tmp_path / "temporal.sqlite"),
            "automatic_capture_enabled": False,
            "embedding_provider": "hash",
            "embedding_dimensions": 64,
            "fact_top_k_planner": 10,
            "projection_candidate_multiplier": 2,
            "semantic_candidate_limit": 20,
            "lexical_candidate_limit": 20,
        }
    )
    return LayeredMemory(config=config, root=tmp_path, llm=FakeLLM())


def _segment(
    memory: LayeredMemory,
    *,
    segment_id: str,
    session_id: str,
    constraint: str,
    created_at: str,
    user_id: str = "user-a",
    project_id: str = "project-a",
) -> MemoryDocument:
    summary = f"数据库技术选型的历史记录：{constraint}"
    return MemoryDocument(
        document_id=segment_id,
        document_type="conversation_segment",
        scope=memory.scope(user_id, project_id=project_id),
        content=f"数据库 技术选型 SQLite Qdrant pgvector。{summary}",
        metadata={
            "session_id": session_id,
            "summary": summary,
            "abstract": summary,
            "constraints": [constraint],
            "decisions": [],
            "user_corrections": [constraint],
            "unresolved": [],
            "active_entities": ["数据库"],
            "detail_ref": {"session_id": session_id},
        },
        created_at=created_at,
        updated_at=created_at,
    )


@pytest.mark.asyncio
async def test_temporal_packet_keeps_archive_but_only_projects_current_state(
    tmp_path: Path,
):
    memory = _memory(tmp_path)
    await memory.open()
    try:
        old = "数据库必须使用 SQLite"
        middle = "数据库必须使用 Qdrant"
        current = "数据库必须使用 pgvector"
        documents = [
            _segment(
                memory,
                segment_id="segment-db-1",
                session_id="session-db",
                constraint=old,
                created_at="2026-01-01T00:00:00+08:00",
            ),
            _segment(
                memory,
                segment_id="segment-db-2",
                session_id="session-db",
                constraint=middle,
                created_at="2026-02-01T00:00:00+08:00",
            ),
            _segment(
                memory,
                segment_id="segment-db-3",
                session_id="session-db",
                constraint=current,
                created_at="2026-03-01T00:00:00+08:00",
            ),
        ]
        await memory.facts.upsert_documents(documents)
        session = SessionMemory(
            user_id="user-a",
            project_id="project-a",
            session_id="session-db",
            rolling_state=RollingMemoryState(
                revision=3,
                segment_ids=[item.document_id for item in documents],
                constraints=[current],
            ),
        )

        packet = await memory.planner_packet(session, "当前数据库技术选型是什么")
        archive = await memory.facts.search_documents(
            "数据库 技术选型 SQLite Qdrant pgvector",
            scope=memory.scope("user-a", project_id="project-a"),
            document_types={"conversation_segment"},
            limit=10,
        )
    finally:
        await memory.close()

    assert {item.document.document_id for item in archive} == {
        "segment-db-1",
        "segment-db-2",
        "segment-db-3",
    }
    assert packet.current_state["authority"] == "session.rolling_state"
    assert packet.current_state["constraints"] == [current]
    assert current in packet.user_constraints
    assert old not in packet.user_constraints
    assert middle not in packet.user_constraints
    assert packet.historical_segments
    assert all(
        item["state_projection_allowed"] is False
        for item in packet.historical_segments
    )
    assert all(not item.get("constraints") for item in packet.historical_segments)
    assert all(not item.get("user_corrections") for item in packet.historical_segments)
    assert all(
        "HISTORICAL EVIDENCE" in item["summary"]
        for item in packet.historical_segments
    )


@pytest.mark.asyncio
async def test_retracted_segment_state_is_not_reactivated(tmp_path: Path):
    memory = _memory(tmp_path)
    await memory.open()
    try:
        old = "所有回答必须使用中文"
        await memory.facts.upsert_document(
            _segment(
                memory,
                segment_id="segment-language-1",
                session_id="session-language",
                constraint=old,
                created_at="2026-01-01T00:00:00+08:00",
            )
        )
        session = SessionMemory(
            user_id="user-a",
            project_id="project-a",
            session_id="session-language",
            rolling_state=RollingMemoryState(
                revision=2,
                segment_ids=["segment-language-1", "segment-language-2"],
                constraints=[],
            ),
        )
        packet = await memory.planner_packet(session, "按当前语言偏好回答")
    finally:
        await memory.close()

    assert packet.current_state["constraints"] == []
    assert old not in packet.user_constraints


@pytest.mark.asyncio
async def test_temporal_packet_keeps_user_and_project_scope_isolated(tmp_path: Path):
    memory = _memory(tmp_path)
    await memory.open()
    try:
        allowed = "数据库必须使用 pgvector"
        forbidden_project = "数据库必须使用 MongoDB"
        forbidden_user = "数据库必须使用 Oracle"
        await memory.facts.upsert_documents(
            [
                _segment(
                    memory,
                    segment_id="segment-allowed",
                    session_id="session-a",
                    constraint=allowed,
                    created_at="2026-03-01T00:00:00+08:00",
                ),
                _segment(
                    memory,
                    segment_id="segment-other-project",
                    session_id="session-b",
                    constraint=forbidden_project,
                    created_at="2026-04-01T00:00:00+08:00",
                    project_id="project-b",
                ),
                _segment(
                    memory,
                    segment_id="segment-other-user",
                    session_id="session-c",
                    constraint=forbidden_user,
                    created_at="2026-05-01T00:00:00+08:00",
                    user_id="user-b",
                ),
            ]
        )
        session = SessionMemory(
            user_id="user-a",
            project_id="project-a",
            session_id="session-a",
            rolling_state=RollingMemoryState(
                revision=1,
                segment_ids=["segment-allowed"],
                constraints=[allowed],
            ),
        )
        packet = await memory.planner_packet(session, "当前数据库技术选型")
    finally:
        await memory.close()

    serialized = packet.model_dump_json()
    assert allowed in serialized
    assert forbidden_project not in serialized
    assert forbidden_user not in serialized
