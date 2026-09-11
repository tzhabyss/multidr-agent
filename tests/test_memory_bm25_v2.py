from __future__ import annotations

from pathlib import Path

from deepresearch.memory_system.embeddings import HashEmbeddingProvider
from deepresearch.memory_system.models import CreateFact, MemoryDocument, MemoryScope
from deepresearch.memory_system.repository import MemoryRepository
from deepresearch.memory_system.retrieval import fts_query_expression, lexical_index_text


def _drain(repository: MemoryRepository, provider: HashEmbeddingProvider) -> None:
    for _ in range(10):
        repository.process_index_outbox(provider, limit=100)
        repository.process_document_index_outbox(provider, limit=100)
        stats = repository.stats()
        if stats.pending_indexes == 0 and stats.pending_document_indexes == 0:
            return
    raise AssertionError("index outbox did not drain")


def test_chinese_lexical_features_are_symmetric_and_searchable(tmp_path: Path):
    assert "检索" in lexical_index_text("混合检索候选窗口")
    assert '"检索"' in fts_query_expression("检索候选窗口是多少")

    repository = MemoryRepository(tmp_path / "bm25.sqlite")
    repository.initialize()
    provider = HashEmbeddingProvider(64)
    scope = MemoryScope(user_id="user-a", project_id="project-a")
    fact = repository.create(
        CreateFact(
            scope=scope,
            content="混合检索的候选窗口为50条",
            category="constraint",
        )
    )
    repository.upsert_document(
        MemoryDocument(
            document_id="episode-chinese",
            document_type="research_episode",
            scope=scope,
            content="上次研究确认中文候选召回需要统一分词",
        )
    )
    _drain(repository, provider)

    fact_rows = repository.retrieval_rows(
        scope,
        "检索候选窗口是多少",
        candidate_ids=[],
        lexical_limit=10,
        limit=10,
    )
    assert any(item.fact_id == fact.fact_id and score > 0 for item, _, _, score in fact_rows)
    document_rows = repository.document_candidates(
        scope,
        "中文候选召回怎么分词",
        candidate_ids=[],
        lexical_limit=10,
        limit=10,
    )
    assert any(item.document_id == "episode-chinese" and score > 0 for item, score in document_rows)


def test_legacy_fts_projection_is_migrated_and_rebuilt(tmp_path: Path):
    repository = MemoryRepository(tmp_path / "migration.sqlite")
    repository.initialize()
    provider = HashEmbeddingProvider(64)
    scope = MemoryScope(user_id="user-a", project_id="project-a")
    fact = repository.create(
        CreateFact(scope=scope, content="用户偏好中文详细回答")
    )
    repository.upsert_document(
        MemoryDocument(
            document_id="segment-legacy",
            document_type="conversation_segment",
            scope=scope,
            content="历史对话记录了中文回答偏好",
        )
    )
    _drain(repository, provider)

    with repository.connection() as connection:
        connection.execute("DROP TABLE memory_fts")
        connection.execute(
            "CREATE VIRTUAL TABLE memory_fts USING fts5(fact_id UNINDEXED, content, category)"
        )
        connection.execute(
            "INSERT INTO memory_fts(fact_id, content, category) VALUES (?, ?, ?)",
            (fact.fact_id, fact.content, fact.category),
        )
        connection.execute("DROP TABLE memory_document_fts")
        connection.execute(
            """CREATE VIRTUAL TABLE memory_document_fts USING fts5(
                   document_id UNINDEXED, content, document_type UNINDEXED
               )"""
        )
        connection.execute(
            """INSERT INTO memory_document_fts(document_id, content, document_type)
               VALUES (?, ?, ?)""",
            ("segment-legacy", "历史对话记录了中文回答偏好", "conversation_segment"),
        )

    repository.initialize()
    with repository.connection() as connection:
        fact_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(memory_fts)")
        }
        document_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(memory_document_fts)")
        }
    assert "search_text" in fact_columns
    assert "search_text" in document_columns
    assert repository.stats().pending_indexes == 1
    assert repository.stats().pending_document_indexes == 1

    _drain(repository, provider)
    rows = repository.retrieval_rows(
        scope,
        "中文详细回答",
        candidate_ids=[],
        lexical_limit=10,
        limit=10,
    )
    assert rows[0][0].fact_id == fact.fact_id
    assert rows[0][3] > 0
