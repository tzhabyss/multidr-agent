from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deepresearch.mcp import (
    MCPClientManager,
    MCPConfigurationError,
    MCPRemoteTool,
    MCPToolCallError,
    _resolve_stdio_command,
)
from deepresearch.runtime_config import (
    MCPRuntimeConfig,
    load_runtime_config,
    resolve_runtime_config_path,
)
from deepresearch.skill_registry import SkillRegistry


class _Text:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    isError = False

    def __init__(self, *, structured_content=None, content=None) -> None:
        self.structuredContent = structured_content
        self.content = content or []


class _FakeSession:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, *, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return self.result


def _config(*, result_kind: str = "search_hits") -> MCPRuntimeConfig:
    return MCPRuntimeConfig.model_validate(
        {
            "enabled": True,
            "servers": {
                "academic": {
                    "command": "python",
                    "tools": {
                        "search_papers": {
                            "skill_id": "academic-search",
                            "result_kind": result_kind,
                            "roles": ["worker"],
                            "profiles": ["general_research"],
                        }
                    },
                }
            },
        }
    )


def _manager(session: _FakeSession, *, result_kind: str = "search_hits") -> MCPClientManager:
    config = _config(result_kind=result_kind)
    manager = MCPClientManager(config)
    manager._connections["academic"] = SimpleNamespace(
        config=config.servers["academic"],
        session=session,
        tools={
            "search_papers": MCPRemoteTool(
                name="search_papers",
                description="Search academic papers",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            "write_paper": MCPRemoteTool(
                name="write_paper",
                description="Must never be exposed without an allowlist entry",
                input_schema={"type": "object", "properties": {}},
            ),
        },
    )
    return manager


def test_stdio_command_uses_active_python_interpreter():
    import sys

    assert _resolve_stdio_command("python") == sys.executable
    assert _resolve_stdio_command("python.exe") == sys.executable


def test_stdio_command_removes_windows_shim_suffix_off_windows(monkeypatch):
    monkeypatch.setattr("deepresearch.mcp.os.name", "posix")

    assert _resolve_stdio_command("npx.cmd") == "npx"
    assert _resolve_stdio_command("custom.bat") == "custom"


@pytest.mark.asyncio
async def test_mcp_transport_is_opened_and_closed_in_the_same_task(monkeypatch):
    config = _config()
    manager = MCPClientManager(config)

    class TaskBoundStack:
        def __init__(self):
            self.owner = asyncio.current_task()
            self.closed = False

        async def aclose(self):
            assert asyncio.current_task() is self.owner
            self.closed = True

    stack = None

    async def connect(server_config):
        nonlocal stack
        stack = TaskBoundStack()
        return SimpleNamespace(
            config=server_config,
            session=SimpleNamespace(),
            stack=stack,
            tools={},
        )

    monkeypatch.setattr(manager, "_connect", connect)
    await manager.open()
    await manager.close()

    assert stack is not None and stack.closed


@pytest.mark.asyncio
async def test_mcp_double_close_is_harmless_and_manager_can_reopen(monkeypatch):
    manager = MCPClientManager(_config())
    stacks = []

    class Stack:
        closed = False

        async def aclose(self):
            self.closed = True

    async def connect(server_config):
        stack = Stack()
        stacks.append(stack)
        return SimpleNamespace(
            config=server_config,
            session=_FakeSession(_Result()),
            stack=stack,
            tools={},
        )

    monkeypatch.setattr(manager, "_connect", connect)
    await manager.open()
    await manager.close()
    await manager.close()
    await manager.open()
    await manager.close()

    assert len(stacks) == 2
    assert all(stack.closed for stack in stacks)


@pytest.mark.asyncio
async def test_mcp_owner_shutdown_fails_active_and_pending_calls(monkeypatch):
    manager = MCPClientManager(_config())
    active = asyncio.Event()

    class BlockingSession(_FakeSession):
        async def call_tool(self, *, name: str, arguments: dict):
            active.set()
            await asyncio.Event().wait()

    class Stack:
        async def aclose(self):
            return None

    async def connect(server_config):
        return SimpleNamespace(
            config=server_config,
            session=BlockingSession(_Result()),
            stack=Stack(),
            tools={},
        )

    monkeypatch.setattr(manager, "_connect", connect)
    await manager.open()
    first = asyncio.create_task(manager.call("academic", "search_papers", {"query": "one"}))
    await active.wait()
    second = asyncio.create_task(manager.call("academic", "search_papers", {"query": "two"}))
    await asyncio.sleep(0)
    manager._owner_tasks["academic"].cancel()

    with pytest.raises(MCPToolCallError, match="closed"):
        await first
    with pytest.raises(MCPToolCallError, match="closed"):
        await second
    await manager.close()


@pytest.mark.asyncio
async def test_allowlisted_mcp_search_tool_becomes_role_scoped_skill():
    session = _FakeSession(
        _Result(
            structured_content={
                "results": [
                    {
                        "title": "Evidence paper",
                        "url": "https://example.org/paper",
                        "abstract": "A useful abstract.",
                    }
                ]
            }
        )
    )
    manager = _manager(session)
    definitions = manager.skill_definitions()

    assert [item.skill_id for item in definitions] == ["academic-search"]
    assert definitions[0].result_kind == "search_hits"
    assert "search" in definitions[0].tags
    assert "mcp" in definitions[0].tags

    registry = SkillRegistry()
    registry.register(definitions[0])
    assert registry.to_openai_tools(role="planner", profile_id="general_research") == []
    tools = registry.to_openai_tools(role="worker", profile_id="general_research", tags=("search",))
    assert [item["function"]["name"] for item in tools] == ["academic_search"]

    result = await registry.execute(
        role="worker",
        profile_id="general_research",
        skill_name="academic_search",
        arguments={"query": "retrieval augmented generation"},
    )

    assert session.calls == [("search_papers", {"query": "retrieval augmented generation"})]
    assert result == [
        {
            "title": "Evidence paper",
            "url": "https://example.org/paper",
            "snippet": "A useful abstract.",
            "provider": "mcp:academic",
            "rank": 1,
        }
    ]


@pytest.mark.asyncio
async def test_observation_mcp_tool_is_returned_without_search_result_coercion():
    session = _FakeSession(_Result(content=[_Text("Paper text")]))
    definition = _manager(session, result_kind="observation").skill_definitions()[0]
    registry = SkillRegistry()
    registry.register(definition)

    result = await registry.execute(
        role="worker",
        profile_id="general_research",
        skill_name="academic_search",
        arguments={"query": "paper id"},
    )

    assert result["server"] == "academic"
    assert result["content"] == [{"type": "text", "text": "Paper text"}]


def test_unavailable_allowlisted_remote_tool_is_a_configuration_error():
    manager = _manager(_FakeSession(_Result()))
    manager._connections["academic"].config.tools["missing"] = (
        manager._connections["academic"].config.tools["search_papers"]
    )

    with pytest.raises(MCPConfigurationError, match="does not expose"):
        manager.skill_definitions()


def test_fastmcp_structured_result_wrapper_is_converted_to_search_hits():
    payload = {
        "server": "academic",
        "tool": "search_papers",
        "content": [],
        "structured_content": {
            "result": [
                {
                    "title": "Wrapped result",
                    "url": "https://arxiv.org/abs/1234.5678",
                    "abstract": "Abstract text",
                }
            ]
        },
    }
    assert MCPClientManager._as_search_hits(payload) == [
        {
            "title": "Wrapped result",
            "url": "https://arxiv.org/abs/1234.5678",
            "snippet": "Abstract text",
            "provider": "mcp:academic",
            "rank": 1,
        }
    ]


def test_markdown_sources_are_converted_without_generated_answer():
    payload = {
        "server": "deepseek-web-search",
        "tool": "web_search",
        "content": [
            {
                "type": "text",
                "text": """\
## Search Results Summary

This generated answer must not become a search snippet or evidence.

---

### Sources (3):

1. [Official report](https://example.org/report)
   - *2 days ago*
2. [Independent analysis](https://news.example.net/analysis)
3. [Duplicate](https://example.org/report)
""",
            }
        ],
    }

    assert MCPClientManager._as_search_hits(payload) == [
        {
            "title": "Official report",
            "url": "https://example.org/report",
            "snippet": "",
            "provider": "mcp:deepseek-web-search",
            "rank": 1,
        },
        {
            "title": "Independent analysis",
            "url": "https://news.example.net/analysis",
            "snippet": "",
            "provider": "mcp:deepseek-web-search",
            "rank": 2,
        },
    ]


def test_anysearch_markdown_results_are_converted_and_snippets_are_bounded():
    payload = {
        "server": "anysearch",
        "tool": "search",
        "content": [
            {
                "type": "text",
                "text": """\
## Search Results (2 results, 120ms)

### 1. Primary report
- **URL**: https://example.org/report
- Direct source summary.

### 2. Independent report
- **URL**: https://news.example.net/report
- Independent summary.
""",
            }
        ],
    }

    assert MCPClientManager._as_search_hits(payload) == [
        {
            "title": "Primary report",
            "url": "https://example.org/report",
            "snippet": "- Direct source summary.",
            "provider": "mcp:anysearch",
            "rank": 1,
        },
        {
            "title": "Independent report",
            "url": "https://news.example.net/report",
            "snippet": "- Independent summary.",
            "provider": "mcp:anysearch",
            "rank": 2,
        },
    ]


def test_stdio_environment_reads_secrets_relative_to_config(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    secret_path = tmp_path / "deepseek-key.txt"
    secret_path.write_text("test-secret\n", encoding="utf-8")
    config = MCPRuntimeConfig.model_validate(
        {
            "enabled": True,
            "servers": {
                "deepseek": {
                    "command": "npx",
                    "env": {"WEBSEARCH_MODEL": "deepseek-flash"},
                    "env_files": {"DEEPSEEK_API_KEY": "../deepseek-key.txt"},
                }
            },
        }
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    environment = MCPClientManager(config, base_dir=config_dir)._stdio_environment(
        config.servers["deepseek"]
    )

    assert environment["DEEPSEEK_API_KEY"] == "test-secret"
    assert environment["WEBSEARCH_MODEL"] == "deepseek-flash"


def test_streamable_http_headers_expand_secret_from_env_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    secret_path = tmp_path / "anysearch-key.txt"
    secret_path.write_text("as_sk_test\n", encoding="utf-8")
    config = MCPRuntimeConfig.model_validate(
        {
            "enabled": True,
            "servers": {
                "anysearch": {
                    "transport": "streamable_http",
                    "url": "https://api.anysearch.com/mcp",
                    "env_files": {"ANYSEARCH_API_KEY": "../anysearch-key.txt"},
                    "headers": {
                        "Authorization": "Bearer ${ANYSEARCH_API_KEY}",
                        "X-Anysearch-Client": "test/1.0",
                    },
                }
            },
        }
    )

    headers = MCPClientManager(config, base_dir=config_dir)._http_headers(
        config.servers["anysearch"]
    )

    assert headers == {
        "Authorization": "Bearer as_sk_test",
        "X-Anysearch-Client": "test/1.0",
    }


def test_stdio_environment_rejects_missing_secret_file(tmp_path):
    config = MCPRuntimeConfig.model_validate(
        {
            "enabled": True,
            "servers": {
                "deepseek": {
                    "command": "npx",
                    "env_files": {"DEEPSEEK_API_KEY": "missing.txt"},
                }
            },
        }
    )

    with pytest.raises(MCPConfigurationError, match="Unable to read environment file"):
        MCPClientManager(config, base_dir=tmp_path)._stdio_environment(
            config.servers["deepseek"]
        )


@pytest.mark.asyncio
async def test_bundled_mcp_servers_complete_stdio_handshake_and_expose_allowlist(monkeypatch):
    monkeypatch.setenv("DEEPRESEARCH_ENABLE_RESEARCH_MCP", "1")
    loaded = load_runtime_config()
    config = loaded.model_copy(
        update={"mcp": loaded.mcp.model_copy(update={"enabled": True})}
    ).mcp
    manager = MCPClientManager(
        config,
        base_dir=resolve_runtime_config_path().parent,
    )
    await manager.open()
    try:
        assert "paper-search" not in manager.errors
        assert "github" not in manager.errors
        assert {
            "arxiv-search",
            "scholar-search",
            "openalex-search",
            "github-repository-search",
            "github-code-search",
        }.issubset({item.skill_id for item in manager.skill_definitions()})
        # Empty queries are deterministic and avoid external network calls while
        # still exercising the complete client -> server -> result path.
        result = await manager.call("paper-search", "search_arxiv", {"query": ""})
        assert result["structured_content"] == {"result": []}
        assert MCPClientManager._as_search_hits(result) == []
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_mcp_error_result_is_propagated_to_the_skill_loop():
    result = _Result(content=[_Text("Remote provider rejected the query")])
    result.isError = True
    definition = _manager(_FakeSession(result)).skill_definitions()[0]
    registry = SkillRegistry()
    registry.register(definition)

    with pytest.raises(MCPToolCallError, match="rejected"):
        await registry.execute(
            role="worker",
            profile_id="general_research",
            skill_name="academic_search",
            arguments={"query": "bad request"},
        )
