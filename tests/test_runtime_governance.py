import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepresearch.tool_runtime import (
    ToolRuntime, GovernanceConfig, ToolPolicy, CircuitOpenError,
    bind_runtime, operation_context,
)
from deepresearch.runtime_monitor import RuntimeMonitor
from deepresearch.skill_registry import SkillDefinition, SkillRegistry, SkillError, SkillAuthorizationError
from deepresearch.skill_loader import load_skill_registry
from deepresearch.skill_runtime import SkillRuntimeManager
from deepresearch.service import ResearchService
from deepresearch.mcp import MCPClientManager, _ConnectedMCPServer
from deepresearch.runtime_config import MCPRuntimeConfig
from tests.fakes import FakeLLM, FakeSearchBackend, FakeFetcher


@pytest.mark.asyncio
async def test_cache_is_run_scoped_copied_and_still_authorized():
    runtime = ToolRuntime(GovernanceConfig(policies={"search": ToolPolicy(
        cache_read_only=True, cache_ttl_seconds=10)}))
    calls = []
    async def search():
        calls.append(1)
        return [{"title": "original"}]
    with operation_context(run_id="a"):
        first = await runtime.execute("search", search, arguments={"query": "secret"})
        first[0]["title"] = "changed"
        second = await runtime.execute("search", search, arguments={"query": "secret"})
        assert second[0]["title"] == "original"
        def deny():
            raise PermissionError("secret credential")
        with pytest.raises(PermissionError):
            await runtime.execute("search", search, arguments={"query": "secret"}, preflight=deny)
    with operation_context(run_id="b"):
        await runtime.execute("search", search, arguments={"query": "secret"})
    assert len(calls) == 2
    assert "secret" not in json.dumps(list(runtime.records))
    metrics = RuntimeMonitor(runtime).summary()["metrics"].values()
    assert sum(m["attempts"] for m in metrics) == 2
    assert sum(m["cache_hits"] for m in metrics) == 1


@pytest.mark.asyncio
async def test_breaker_only_one_probe_and_empty_is_not_transport_failure():
    runtime = ToolRuntime(GovernanceConfig(policies={"x": ToolPolicy(
        circuit_enabled=True, failure_threshold=1, recovery_seconds=1)}))
    async def bad():
        raise ConnectionError("down")
    with operation_context(run_id="r"):
        with pytest.raises(ConnectionError):
            await runtime.execute("x", bad)
        with pytest.raises(CircuitOpenError):
            await runtime.execute("x", bad)
        circuit = runtime.circuits[("r", "x", "", "")]
        circuit.opened_until = 1
        started, release = asyncio.Event(), asyncio.Event()
        async def probe():
            started.set()
            await release.wait()
            return []
        task = asyncio.create_task(runtime.execute("x", probe))
        await started.wait()
        with pytest.raises(CircuitOpenError):
            await runtime.execute("x", probe)
        release.set()
        await task
        assert circuit.state() == "closed"
        assert runtime.records[-1]["status"] == "empty"


@pytest.mark.asyncio
async def test_cancel_releases_half_open_probe_and_records_one_terminal():
    runtime = ToolRuntime(GovernanceConfig(policies={"x": ToolPolicy(circuit_enabled=True)}))
    entered = asyncio.Event()
    async def wait():
        entered.set()
        await asyncio.Event().wait()
    with operation_context(run_id="r"):
        task = asyncio.create_task(runtime.execute("x", wait))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(runtime.records) == 1
    assert runtime.records[0]["status"] == "cancelled"
    assert not runtime.circuits[("r", "x", "", "")].probe


@pytest.mark.asyncio
async def test_fallback_preserves_parent_and_failed_child():
    registry = SkillRegistry()
    async def bad():
        raise ConnectionError()
    async def good():
        return ["result"]
    for name, handler, fallback in [("search", None, ("a", "b")), ("a", bad, ()), ("b", good, ())]:
        registry.register(SkillDefinition(name, name, "", frozenset({"worker"}), frozenset(),
                                          handler=handler, fallback_skills=fallback))
    runtime = ToolRuntime()
    with bind_runtime(runtime), operation_context(run_id="r"):
        assert await registry.execute(role="worker", profile_id="p", skill_name="search", arguments={}) == ["result"]
    rows = {r["name"]: r for r in runtime.records}
    assert rows["a"]["status"] == "failed"
    assert rows["b"]["parent_operation_id"] == rows["search"]["operation_id"]


def test_nested_schema_rejects_invalid_array_item():
    skill = SkillDefinition("x", "x", "", frozenset({"worker"}), frozenset(), parameters={
        "type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}
        }}}
    })
    with pytest.raises(SkillError):
        SkillRegistry._validate_arguments(skill, {"items": [{"id": "wrong"}]})


def make_skill_root(tmp_path):
    directory = tmp_path / "skills" / "rule"
    directory.mkdir(parents=True)
    (directory / "skill.json").write_text(json.dumps({"roles": ["worker"], "version": "1.0.0"}))
    path = directory / "SKILL.md"
    path.write_text("---\nname: rule\ndescription: rule\n---\nold instruction")
    return path


@pytest.mark.asyncio
async def test_snapshot_pins_lazy_body_across_concurrent_reload(tmp_path):
    path = make_skill_root(tmp_path)
    registry = load_skill_registry(tmp_path, skill_mode="layered")
    manager = SkillRuntimeManager(registry)
    snapshot = manager.snapshot()
    assert not registry.get("rule").instructions_loaded
    ready, finish = asyncio.Event(), asyncio.Event()
    async def old_run():
        with manager.bind(snapshot):
            ready.set()
            await finish.wait()
            return registry.get("rule").load_instructions()
    task = asyncio.create_task(old_run())
    await ready.wait()
    path.write_text("---\nname: rule\ndescription: rule\n---\nnew instruction")
    assert not manager.reload()["activated"]
    assert manager.reload(approved=True)["activated"]
    with manager.bind(manager.snapshot()):
        assert registry.get("rule").load_instructions() == "new instruction"
    finish.set()
    assert await task == "old instruction"
    manager.rollback()
    assert registry.get("rule").load_instructions() == "old instruction"


def test_bad_reload_retains_valid_skill_and_snapshot_integrity(tmp_path):
    path = make_skill_root(tmp_path)
    registry = load_skill_registry(tmp_path, skill_mode="layered")
    manager = SkillRuntimeManager(registry)
    snapshot = manager.snapshot()
    path.write_text("invalid")
    report = manager.reload(approved=True)
    assert report["errors"] and not report["activated"]
    with manager.bind(snapshot):
        assert registry.get("rule").load_instructions() == "old instruction"
    snapshot["skills"]["rule"]["instructions"] = "tampered"
    with pytest.raises(ValueError, match="integrity"):
        with manager.bind(snapshot):
            pass


@pytest.mark.asyncio
async def test_service_persists_monitor_and_skill_snapshot(tmp_path):
    async with ResearchService.in_memory(llm=FakeLLM(), search_backend=FakeSearchBackend(),
                                        fetcher=FakeFetcher(), artifact_root=tmp_path) as service:
        result = await service.run({"question": "Explain reliable research", "run_id": "observed"})
        assert service.monitor_summary()["metrics"]
        rows = [json.loads(line) for path in (Path(result.artifact_dir) / "details/runtime").glob("operations-*.jsonl")
                for line in path.read_text().splitlines()]
        assert rows and all(r["run_id"] == "observed" for r in rows)
        assert len({r["operation_id"] for r in rows}) == len(rows)
        assert any(r["kind"] == "stage" for r in rows)
        assert all(r.get("task_id") for r in rows if r["kind"] == "tool")
        assert (Path(result.artifact_dir) / "details/runtime/skill-snapshot.json").is_file()
    assert service.monitor.task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_mcp_cancel_does_not_block_following_request(termination):
    config = MCPRuntimeConfig.model_validate({"enabled": True, "servers": {"s": {
        "command": "unused", "tools": {"read": {"timeout_seconds": .1}}
    }}})
    manager = MCPClientManager(config)
    entered, cleaned = asyncio.Event(), asyncio.Event()
    class Session:
        async def call_tool(self, *, name, arguments):
            if arguments.get("hang"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return SimpleNamespace(isError=False, structuredContent={"ok": True}, content=[])
    async def connect(policy):
        return _ConnectedMCPServer(policy, Session(), AsyncExitStack(), {})
    manager._connect = connect
    await manager.open()
    try:
        call = asyncio.create_task(manager.call("s", "read", {"hang": True}))
        await entered.wait()
        if termination == "cancel":
            call.cancel()
        with pytest.raises(asyncio.CancelledError if termination == "cancel" else TimeoutError):
            await call
        await asyncio.wait_for(cleaned.wait(), 1)
        result = await asyncio.wait_for(manager.call("s", "read", {}), 1)
        assert result["structured_content"] == {"ok": True}
    finally:
        await manager.close()


def test_reload_installed_instruction_only_skill_and_rollback(tmp_path):
    make_skill_root(tmp_path)
    registry = load_skill_registry(tmp_path, skill_mode="layered")
    manager = SkillRuntimeManager(registry)
    added = tmp_path / "skills" / "new-rule"
    added.mkdir()
    (added / "SKILL.md").write_text("---\nname: new-rule\ndescription: new rule\n---\nnew body")
    (added / "skill.json").write_text(json.dumps({"roles": ["worker"]}))
    assert manager.reload()["changed"] == ["new-rule"]
    assert registry.get("new-rule") is None
    assert manager.reload(approved=True)["activated"]
    snapshot = manager.snapshot()
    with manager.bind(snapshot):
        assert registry.get("new-rule").load_instructions() == "new body"
    manager.rollback()
    assert registry.get("new-rule") is None


def test_schema_forbids_external_references():
    with pytest.raises(SkillError):
        SkillRegistry._validate_arguments_schema({"$ref": "https://example.com/schema"})


@pytest.mark.asyncio
async def test_expired_cache_and_skill_alias_policy():
    registry = SkillRegistry()
    calls = []
    async def handler():
        calls.append(1)
        return ["ok"]
    registry.register(SkillDefinition(skill_id="test-search", description="search", instructions="",
        roles=frozenset({"worker"}), profiles=frozenset(), handler=handler))
    runtime = ToolRuntime(GovernanceConfig(policies={"test-search": ToolPolicy(
        cache_read_only=True, cache_ttl_seconds=10)}))
    with bind_runtime(runtime), operation_context(run_id="cache"):
        for _ in range(2):
            await registry.execute(role="worker", profile_id="default", skill_name="test_search", arguments={})
        assert len(calls) == 1
        key = next(iter(runtime.cache))
        runtime.cache[key] = (0, runtime.cache[key][1])
        await registry.execute(role="worker", profile_id="default", skill_name="test-search", arguments={})
        assert len(calls) == 2
    assert all(r["name"] == "test-search" for r in runtime.records)
