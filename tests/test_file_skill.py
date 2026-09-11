from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.file_tools import LocalFileReader, LocalFileSecurityError
from deepresearch.evidence import records_from_document
from deepresearch.agent_loop import AgentLoop
from deepresearch.llm import LLMResponse, LLMToolCall
from deepresearch.runtime_config import load_runtime_config
from deepresearch.retrieval import independent_domain
from deepresearch.schemas import FetchedDocument
from deepresearch.schemas import ResearchTask, TaskBudget
from deepresearch.skill_loader import load_skill_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_file_reader_reads_text_json_and_csv(tmp_path: Path):
    text_path = tmp_path / "notes.md"
    json_path = tmp_path / "data.json"
    csv_path = tmp_path / "table.csv"
    tsv_path = tmp_path / "table.tsv"
    text_path.write_text("Research note", encoding="utf-8")
    json_path.write_text('{"answer": "42"}', encoding="utf-8")
    csv_path.write_text("name,value\nalpha,1\n", encoding="utf-8")
    tsv_path.write_text("name\tvalue\nalpha\t1\n", encoding="utf-8")

    reader = LocalFileReader(roots=[str(tmp_path)])
    assert reader.read(str(text_path))["content"] == "Research note"
    assert '"answer": "42"' in reader.read(str(json_path))["content"]
    assert "name\tvalue" in reader.read(str(csv_path))["content"]
    assert "name\tvalue" in reader.read(str(tsv_path))["content"]
    assert reader.read(str(text_path))["final_url"].startswith("file://")


def test_file_reader_rejects_traversal_oversize_and_unsupported_files(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    large = tmp_path / "large.txt"
    large.write_text("123456", encoding="utf-8")
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"bin")
    reader = LocalFileReader(roots=[str(tmp_path)], max_file_bytes=5)

    with pytest.raises(LocalFileSecurityError, match="outside"):
        reader.read(str(outside))
    with pytest.raises(LocalFileSecurityError, match="exceeds"):
        reader.read(str(large))
    with pytest.raises(ValueError, match="unsupported"):
        reader.read(str(binary))


def test_file_reader_rejects_symlink_when_platform_allows_creation(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(LocalFileSecurityError, match="symlink"):
        LocalFileReader(roots=[str(tmp_path)]).read(str(link))


def test_file_skill_is_worker_visible_alongside_search_but_not_safe_fetch():
    registry = load_skill_registry(PROJECT_ROOT)
    tools = registry.to_openai_tools(
        role="worker",
        profile_id="general_research",
        any_tags=("search", "file"),
    )
    names = {item["function"]["name"] for item in tools}
    assert "file_reader" in names
    assert "web_search" in names
    assert "safe_fetch" not in names


def test_both_profiles_expose_hardened_fetch_with_evaluation_alias_isolated():
    registry = load_skill_registry(PROJECT_ROOT)
    general_fetchers = {
        item.skill_id
        for item in registry.list(role="worker", profile_id="general_research", tags=("fetch",))
    }
    evaluation_fetchers = {
        item.skill_id
        for item in registry.list(role="worker", profile_id="drb_evaluation", tags=("fetch",))
    }

    assert "safe-fetch-evaluation" not in general_fetchers
    assert "safe-fetch-evaluation" in evaluation_fetchers
    general_fetch = registry.get("safe-fetch")
    evaluation_fetch = registry.get("safe-fetch-evaluation")
    assert general_fetch is not None and general_fetch.handler is not None
    assert evaluation_fetch is not None and evaluation_fetch.handler is not None
    assert type(general_fetch.handler.__self__).__name__ == "EvaluationWebFetcher"
    assert type(evaluation_fetch.handler.__self__).__name__ == "EvaluationWebFetcher"


def test_local_document_enters_evidence_ledger_as_local_source():
    document = FetchedDocument(
        requested_url="notes.md",
        final_url="file:///workspace/notes.md",
        title="notes.md",
        content="The project uses a local evidence file.",
        content_type="text/markdown",
        http_status=200,
    )
    source, snapshot, evidence, claims, links = records_from_document(
        document,
        "local_file",
        [{"text": "The project uses a local evidence file.", "quote": "local evidence file"}],
    )
    assert source.source_kind == "local_document"
    assert source.quality_tier == 3
    assert independent_domain(source.canonical_uri) == "file:/workspace/notes.md"
    assert source.provenance["independent_domain"] == "file:/workspace/notes.md"
    assert snapshot.content == document.content
    assert evidence[0].location["kind"] == "local_text"
    assert claims[0].status == "supported"
    assert links[0].relation == "supports"


@pytest.mark.asyncio
async def test_agent_loop_can_execute_file_reader(tmp_path: Path, monkeypatch):
    path = tmp_path / "research.txt"
    path.write_text("The local document states a 42 percent increase.", encoding="utf-8")
    monkeypatch.setenv("DEEPRESEARCH_FILE_ROOTS", str(tmp_path))
    registry = load_skill_registry(PROJECT_ROOT)

    class FileCallingLLM:
        model_id = "file-fixture"

        def __init__(self):
            self.calls = 0

        async def complete_with_tools(self, *, messages, tools, max_tokens=1200):
            self.calls += 1
            if self.calls == 1:
                assert any(item["function"]["name"] == "file_reader" for item in tools)
                return LLMResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            call_id="read-1",
                            name="file_reader",
                            arguments={"path": str(path)},
                        )
                    ],
                )
            assert "42 percent" in messages[-1]["content"]
            return LLMResponse(content="The file was read.", finish_reason="stop")

    result = await AgentLoop(
        llm=FileCallingLLM(),
        registry=registry,
        role="worker",
        profile_id="general_research",
        max_iterations=2,
        max_tool_calls=2,
    ).run(
        system_prompt="Read the local file.",
        user_prompt="Read the requested file.",
        any_tags=("search", "file"),
        max_tokens=500,
    )
    assert result.stop_reason == "model_finished"
    assert len(result.executions) == 1
    assert result.executions[0].skill_name == "file_reader"
    assert result.executions[0].result["content"].startswith("The local document")


@pytest.mark.asyncio
async def test_worker_search_loop_records_local_file_as_search_and_candidate(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "worker.txt"
    path.write_text("Worker local source", encoding="utf-8")
    monkeypatch.setenv("DEEPRESEARCH_FILE_ROOTS", str(tmp_path))
    registry = load_skill_registry(PROJECT_ROOT)

    class LocalLookupLLM:
        model_id = "local-loop-fixture"

        def __init__(self):
            self.calls = 0

        async def complete_with_tools(self, *, messages, tools, max_tokens=1200):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            call_id="local-read",
                            name="file_reader",
                            arguments={"path": str(path)},
                        )
                    ],
                )
            return LLMResponse(content="done", finish_reason="stop")

    from deepresearch.agents import ResearchWorker

    worker = ResearchWorker(
        LocalLookupLLM(),
        registry,
        agent_loop_llm=LocalLookupLLM(),
    )
    task = ResearchTask(
        task_id="local-task",
        objective="Read local source",
        research_dimension="local",
        queries=[],
        budget=TaskBudget(max_searches=2, max_fetches=2, max_wall_time_seconds=60),
    )
    result = await worker._search_with_loop(
        {
            "run_id": "local-run",
            "scope": {"options": {"max_worker_iterations": 2}},
            "search_traces": [],
            "claims": [],
            "unresolved_gaps": [],
            "worker_memory": {},
            "current_tasks": [],
        },
        task,
        profile_id="general_research",
        search_index=0,
        llm_index=0,
    )
    candidate_groups, local_documents, search_traces, usage_events, *_ = result
    assert list(local_documents.values())[0].content == "Worker local source"
    assert candidate_groups[0][0][1].provider == "local_file"
    assert search_traces[0]["query"] == str(path)
    assert any(item["category"] == "search" for item in usage_events)


def test_research_mcps_are_env_gated_but_default_search_fanout_is_enabled(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_ENABLE_RESEARCH_MCP", raising=False)
    config = load_runtime_config()
    assert config.mcp.enabled is True
    assert config.mcp.servers["deepseek-web-search"].runtime_enabled()
    assert config.mcp.servers["anysearch"].runtime_enabled()
    assert not config.mcp.servers["paper-search"].runtime_enabled()
    assert not config.mcp.servers["github"].runtime_enabled()

    monkeypatch.setenv("DEEPRESEARCH_ENABLE_RESEARCH_MCP", "1")
    assert config.mcp.servers["paper-search"].runtime_enabled()
    assert config.mcp.servers["github"].runtime_enabled()
    assert config.mcp.servers["deepseek-web-search"].runtime_enabled()
