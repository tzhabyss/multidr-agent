"""MCP client lifecycle and explicit MCP-tool to SkillRegistry adaptation.

MCP servers are external capability providers.  Their tools are never exposed
automatically: only tools declared in RuntimeConfig.mcp.servers.<id>.tools are
adapted to role-scoped SkillDefinition instances.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtime_config import MCPRuntimeConfig, MCPServerRuntimeConfig
from .skill_registry import SkillDefinition


logger = logging.getLogger(__name__)


class MCPError(RuntimeError):
    pass


class MCPConfigurationError(MCPError):
    pass


class MCPAuthorizationError(MCPError):
    pass


class MCPToolCallError(MCPError):
    pass


@dataclass(frozen=True)
class MCPRemoteTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class _ConnectedMCPServer:
    config: MCPServerRuntimeConfig
    session: Any
    stack: AsyncExitStack
    tools: dict[str, MCPRemoteTool]
    owner_task: asyncio.Task | None = None
    requests: asyncio.Queue["_MCPCallRequest | None"] | None = None


@dataclass
class _MCPCallRequest:
    remote_name: str
    arguments: dict[str, Any]
    result: asyncio.Future
    deadline: float = 0


def _expand_environment(
    value: str, environment: dict[str, str] | None = None
) -> str:
    """Expand ${NAME}; fail early rather than silently sending empty secrets."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = (environment or os.environ).get(name)
        if resolved is None:
            raise MCPConfigurationError(
                f"Required environment variable {name!r} is not set for an MCP server"
            )
        return resolved

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)


def _normalise_skill_id(server_id: str, tool_name: str, configured: str | None) -> str:
    raw = configured or f"mcp-{server_id}-{tool_name}"
    normalised = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    if not normalised:
        raise MCPConfigurationError(
            f"MCP tool {server_id}.{tool_name} produced an empty Skill id"
        )
    if normalised[0].isdigit():
        normalised = f"mcp-{normalised}"
    return normalised


def _resolve_stdio_command(command: str) -> str:
    """Resolve portable MCP commands without relying on the caller's shell.

    Windows package-manager shims use ``.cmd``/``.bat`` suffixes, while their
    macOS and Linux counterparts do not.  Bundled Python MCP servers must also
    run inside the active virtual environment even when the CLI was invoked by
    an absolute interpreter path and the environment was not activated.
    """

    resolved = command.strip()
    if resolved.lower() in {"python", "python.exe"}:
        return sys.executable
    if os.name != "nt" and resolved.lower().endswith((".cmd", ".bat")):
        return resolved.rsplit(".", 1)[0]
    return resolved


class MCPClientManager:
    """Owns MCP connections for one ResearchService lifecycle."""

    def __init__(
        self,
        config: MCPRuntimeConfig,
        *,
        base_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.base_dir = Path(base_dir or Path.cwd()).expanduser().resolve()
        self._connections: dict[str, _ConnectedMCPServer] = {}
        self._owner_tasks: dict[str, asyncio.Task] = {}
        self._owner_ready: dict[str, asyncio.Future] = {}
        self._close_task: asyncio.Task | None = None
        self.errors: dict[str, str] = {}

    def _stdio_environment(self, config: MCPServerRuntimeConfig) -> dict[str, str]:
        environment = {
            **os.environ,
            **{key: _expand_environment(value) for key, value in config.env.items()},
        }
        for key, configured_path in config.env_files.items():
            path = Path(_expand_environment(configured_path)).expanduser()
            if not path.is_absolute():
                path = self.base_dir / path
            try:
                value = path.resolve().read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise MCPConfigurationError(
                    f"Unable to read environment file for {key!r}: {path}"
                ) from exc
            if not value:
                raise MCPConfigurationError(
                    f"Environment file for {key!r} is empty: {path}"
                )
            environment[key] = value
        return environment

    def _http_headers(self, config: MCPServerRuntimeConfig) -> dict[str, str]:
        environment = self._stdio_environment(config)
        return {
            key: _expand_environment(value, environment)
            for key, value in config.headers.items()
        }

    async def open(self) -> None:
        if not self.config.enabled:
            return
        if self._close_task is not None:
            # A manager can be reused after a clean close. Do not retain the
            # completed close task, otherwise a later close would be a no-op.
            if not self._close_task.done():
                await asyncio.shield(self._close_task)
            self._close_task = None
        for server_id, server_config in self.config.servers.items():
            if not server_config.runtime_enabled():
                continue
            ready = asyncio.get_running_loop().create_future()
            owner = asyncio.create_task(
                self._connection_owner(server_id, server_config, ready),
                name=f"mcp-owner:{server_id}",
            )
            self._owner_tasks[server_id] = owner
            self._owner_ready[server_id] = ready
            try:
                # The owner task enters, services, and exits the AnyIO
                # transport context. ``shield`` prevents a timeout here from
                # cancelling that task midway through context-manager setup.
                async with asyncio.timeout(server_config.connect_timeout_seconds):
                    connection = await asyncio.shield(ready)
                self._connections[server_id] = connection
                logger.info(
                    "Connected MCP server %s with %d discovered tools",
                    server_id,
                    len(connection.tools),
                )
            except BaseException as exc:
                owner.cancel()
                await self._await_task(owner)
                self._owner_tasks.pop(server_id, None)
                self._owner_ready.pop(server_id, None)
                if isinstance(exc, asyncio.CancelledError):
                    await self.close()
                    raise
                message = f"{type(exc).__name__}: {exc}"
                self.errors[server_id] = message
                logger.warning("MCP server %s is unavailable: %s", server_id, message)
                if server_config.required:
                    await self.close()
                    raise MCPConfigurationError(
                        f"Required MCP server {server_id!r} could not be connected: {message}"
                    ) from exc

    async def close(self) -> None:
        """Stop all owner tasks; transport contexts close in their owner task."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(), name="mcp-manager-close"
            )
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            # Finish cleanup even when the service task is being cancelled,
            # then propagate the original cancellation to its caller.
            await asyncio.shield(close_task)
            raise

    async def _close_impl(self) -> None:
        connections = list(self._connections.values())
        connected_ids = set(self._connections)
        self._connections.clear()
        for connection in reversed(connections):
            if connection.requests is not None:
                await connection.requests.put(None)

        tasks = list(self._owner_tasks.values())
        for server_id, task in list(self._owner_tasks.items()):
            # A task still connecting has no queue yet and must be cancelled;
            # it will close any partially entered stack in its own finally.
            if server_id not in connected_ids and not task.done():
                task.cancel()
        for task in tasks:
            await self._await_task(task)
        self._owner_tasks.clear()
        self._owner_ready.clear()

    @staticmethod
    async def _await_task(task: asyncio.Task) -> None:
        if task.done():
            try:
                task.result()
            except BaseException:
                pass
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cancelled owner still needs to run its finally block.
            if not task.done():
                await asyncio.shield(task)
        except BaseException:
            pass

    async def _connection_owner(
        self,
        server_id: str,
        config: MCPServerRuntimeConfig,
        ready: asyncio.Future,
    ) -> None:
        connection: _ConnectedMCPServer | None = None
        requests: asyncio.Queue[_MCPCallRequest | None] = asyncio.Queue()
        active_request: _MCPCallRequest | None = None
        try:
            async with asyncio.timeout(config.connect_timeout_seconds):
                connection = await self._connect(config)
            connection.owner_task = asyncio.current_task()
            connection.requests = requests
            if not ready.done():
                ready.set_result(connection)
            while True:
                request = await requests.get()
                if request is None:
                    break
                active_request = request
                if request.result.cancelled():
                    active_request = None
                    continue
                try:
                    try:
                        remaining = request.deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise TimeoutError("MCP request expired in queue")
                        # Timeout the actual transport call, not only its waiting caller.
                        remote_task = asyncio.create_task(connection.session.call_tool(
                            name=request.remote_name, arguments=request.arguments,
                        ))
                        try:
                            done, _ = await asyncio.wait(
                                {remote_task, request.result}, timeout=remaining,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if request.result.cancelled():
                                continue
                            if remote_task not in done:
                                raise TimeoutError("MCP transport deadline reached")
                            raw = remote_task.result()
                        finally:
                            if not remote_task.done():
                                remote_task.cancel()
                            await asyncio.gather(remote_task, return_exceptions=True)
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        raise MCPToolCallError(
                            f"MCP tool {server_id}.{request.remote_name} failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    value = self._normalise_result(
                        server_id, request.remote_name, raw
                    )
                except BaseException as exc:
                    if not request.result.done():
                        if isinstance(exc, asyncio.CancelledError):
                            request.result.set_exception(
                                MCPToolCallError(
                                    f"MCP server {server_id!r} closed while a tool call was active"
                                )
                            )
                        else:
                            request.result.set_exception(exc)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                else:
                    if not request.result.done():
                        request.result.set_result(value)
                finally:
                    active_request = None
        except BaseException as exc:
            if not ready.done():
                if isinstance(exc, asyncio.CancelledError):
                    ready.cancel()
                else:
                    ready.set_exception(exc)
        finally:
            shutdown_error = MCPToolCallError(
                f"MCP server {server_id!r} closed before the tool call completed"
            )
            if active_request is not None and not active_request.result.done():
                active_request.result.set_exception(shutdown_error)
            while True:
                try:
                    pending = requests.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if pending is not None and not pending.result.done():
                    pending.result.set_exception(shutdown_error)
            current = self._connections.get(server_id)
            if current is connection:
                self._connections.pop(server_id, None)
            if connection is not None:
                try:
                    await connection.stack.aclose()
                except BaseException as exc:  # pragma: no cover - external transport
                    logger.warning(
                        "Failed to close MCP server %s cleanly: %s", server_id, exc
                    )

    def skill_definitions(self) -> list[SkillDefinition]:
        """Return Skills for connected, explicitly allowlisted remote tools only."""

        if not self.config.enabled:
            return []
        definitions: list[SkillDefinition] = []
        used_ids: set[str] = set()
        for server_id, connection in self._connections.items():
            for remote_name, policy in connection.config.tools.items():
                if not policy.enabled:
                    continue
                remote = connection.tools.get(remote_name)
                if remote is None:
                    raise MCPConfigurationError(
                        f"MCP server {server_id!r} does not expose allowlisted tool {remote_name!r}"
                    )
                skill_id = _normalise_skill_id(server_id, remote_name, policy.skill_id)
                if skill_id in used_ids:
                    raise MCPConfigurationError(
                        f"Duplicate MCP Skill id after normalisation: {skill_id!r}"
                    )
                used_ids.add(skill_id)

                def make_handler(
                    configured_server_id: str,
                    configured_remote_name: str,
                    configured_result_kind: str,
                ):
                    async def handler(**arguments: Any) -> Any:
                        result = await self.call(
                            configured_server_id,
                            configured_remote_name,
                            arguments,
                        )
                        if configured_result_kind == "search_hits":
                            return self._as_search_hits(result)
                        return result

                    return handler

                handler = make_handler(server_id, remote_name, policy.result_kind)

                definitions.append(
                    SkillDefinition(
                        skill_id=skill_id,
                        description=policy.description
                        or remote.description
                        or f"MCP tool {remote_name} from {server_id}",
                        instructions=(
                            f"External MCP tool from server {server_id}. Use only when its "
                            "declared parameters directly help the delegated research task."
                        ),
                        roles=frozenset(policy.roles),
                        profiles=frozenset(policy.profiles),
                        parameters=remote.input_schema,
                        handler=handler,
                        timeout_seconds=policy.timeout_seconds,
                        tags=frozenset(
                            {
                                "mcp",
                                *policy.tags,
                                *({"search"} if policy.result_kind == "search_hits" else set()),
                            }
                        ),
                        source_path=f"mcp://{server_id}/{remote_name}",
                        skill_layer="access",
                        # Remote tools are atomic runtime capabilities, but
                        # phase one only observes them. Candidate generation is
                        # still disabled globally until phase two exists.
                        capability_granularity="atomic",
                        skill_version="mcp-runtime",
                        required_tools=frozenset({remote_name}),
                        output_artifact_roles=frozenset(
                            {"source_document"}
                            if policy.result_kind == "search_hits"
                            else set()
                        ),
                        result_kind=policy.result_kind,
                    )
                )
        return definitions

    async def call(
        self,
        server_id: str,
        remote_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        connection = self._connections.get(server_id)
        if connection is None:
            raise MCPToolCallError(f"MCP server {server_id!r} is not connected")
        policy = connection.config.tools.get(remote_name)
        if policy is None or not policy.enabled:
            raise MCPAuthorizationError(
                f"MCP tool {server_id}.{remote_name} is not allowlisted"
            )
        requests = getattr(connection, "requests", None)
        owner_task = getattr(connection, "owner_task", None)
        if requests is not None and owner_task is not None:
            result_future = asyncio.get_running_loop().create_future()
            await requests.put(
                _MCPCallRequest(remote_name, arguments, result_future,
                                asyncio.get_running_loop().time() + policy.timeout_seconds)
            )
            try:
                return await asyncio.wait_for(result_future, policy.timeout_seconds)
            except asyncio.CancelledError:
                result_future.cancel()
                raise
        try:
            result = await asyncio.wait_for(connection.session.call_tool(
                name=remote_name,
                arguments=arguments,
            ), timeout=policy.timeout_seconds)
        except Exception as exc:
            raise MCPToolCallError(
                f"MCP tool {server_id}.{remote_name} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalise_result(server_id, remote_name, result)

    async def _connect(self, config: MCPServerRuntimeConfig) -> _ConnectedMCPServer:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:  # pragma: no cover - exercised only without optional install
            raise MCPConfigurationError(
                "MCP support requires the optional dependency. Install with: pip install -e .[mcp]"
            ) from exc

        stack = AsyncExitStack()
        try:
            if config.transport == "stdio":
                environment = self._stdio_environment(config)
                read, write = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=_resolve_stdio_command(config.command or ""),
                            args=config.args,
                            env=environment,
                        )
                    )
                )
            else:
                try:
                    from mcp.client.streamable_http import streamablehttp_client
                except ImportError as exc:  # pragma: no cover - depends on installed MCP version
                    raise MCPConfigurationError(
                        "Installed MCP package does not support streamable_http transport"
                    ) from exc
                headers = self._http_headers(config)
                transport = await stack.enter_async_context(
                    streamablehttp_client(config.url or "", headers=headers or None)
                )
                read, write = transport[0], transport[1]

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            response = await session.list_tools()
            tools = {
                str(tool.name): MCPRemoteTool(
                    name=str(tool.name),
                    description=str(getattr(tool, "description", "") or ""),
                    input_schema=dict(
                        getattr(tool, "inputSchema", None)
                        or getattr(tool, "input_schema", None)
                        or {"type": "object", "properties": {}}
                    ),
                )
                for tool in response.tools
            }
            return _ConnectedMCPServer(
                config=config,
                session=session,
                stack=stack,
                tools=tools,
            )
        except BaseException:
            # Cancellation can happen after the transport context has been
            # entered. Close it here, in this same owner task, before letting
            # the cancellation escape; otherwise asyncgen finalization may
            # run in an unrelated task and violate AnyIO cancel-scope rules.
            await stack.aclose()
            raise

    @staticmethod
    def _normalise_result(server_id: str, remote_name: str, result: Any) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for item in list(getattr(result, "content", []) or []):
            item_type = str(getattr(item, "type", "unknown"))
            if item_type == "text":
                content.append({"type": "text", "text": str(getattr(item, "text", ""))})
            elif item_type == "resource":
                resource = getattr(item, "resource", None)
                content.append(
                    {
                        "type": "resource",
                        "uri": str(getattr(resource, "uri", "")),
                        "text": str(getattr(resource, "text", "")),
                    }
                )
            else:
                content.append({"type": item_type, "summary": str(item)[:500]})

        if bool(getattr(result, "isError", getattr(result, "is_error", False))):
            message = "\n".join(item.get("text", item.get("summary", "")) for item in content)
            raise MCPToolCallError(message or f"MCP tool {server_id}.{remote_name} returned an error")

        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        payload: dict[str, Any] = {
            "server": server_id,
            "tool": remote_name,
            "content": content,
        }
        if structured is not None:
            payload["structured_content"] = structured
        return payload

    @staticmethod
    def _as_search_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Map conventional MCP structured search results to the local SearchHit contract."""

        def rows(value: Any) -> list[Any]:
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for key in ("results", "items", "result", "data"):
                    nested = value.get(key)
                    if isinstance(nested, list):
                        return nested
                    if isinstance(nested, dict):
                        nested_rows = rows(nested)
                        if nested_rows:
                            return nested_rows
            return []

        candidates: Any = payload.get("structured_content")
        candidates = rows(candidates)
        if not candidates:
            candidates = []
            for item in payload.get("content", []):
                if item.get("type") != "text":
                    continue
                try:
                    import json

                    decoded = json.loads(item.get("text", ""))
                except (TypeError, ValueError):
                    decoded = None
                if decoded is not None:
                    candidates.extend(rows(decoded))
                    continue

                # Some search MCP servers return a generated answer followed by
                # a Markdown Sources section. Only ingest that source list; the
                # generated answer is an observation, not auditable evidence.
                text = str(item.get("text", ""))
                source_header = re.search(
                    r"(?im)^#{1,6}\s+Sources(?:\s*\(\d+\))?\s*:?\s*$",
                    text,
                )
                if source_header is None:
                    continue
                source_text = text[source_header.end() :]
                for line in source_text.splitlines():
                    match = re.match(
                        r"^\s*\d+[.)]\s+\[([^\]]+)\]\((https?://.+)\)\s*$",
                        line,
                    )
                    if match:
                        candidates.append(
                            {"title": match.group(1).strip(), "url": match.group(2).strip()}
                        )

        if not candidates:
            # AnySearch returns compact Markdown result blocks rather than
            # structuredContent. Parse only the explicit numbered result
            # headings and URL field; arbitrary prose remains an observation.
            for content_item in payload.get("content", []):
                if content_item.get("type") != "text":
                    continue
                text = str(content_item.get("text", ""))
                for match in re.finditer(
                    r"(?ms)^###\s+\d+\.\s+([^\r\n]+)\r?\n"
                    r"(.*?)(?=^###\s+\d+\.|^##\s+Query\s+\d+:|\Z)",
                    text,
                ):
                    title = match.group(1).strip()
                    body = match.group(2)
                    url_match = re.search(
                        r"(?im)^-\s+\*\*URL\*\*:\s*(https?://\S+)\s*$",
                        body,
                    )
                    if url_match is None:
                        continue
                    snippet = (
                        body[: url_match.start()] + body[url_match.end() :]
                    ).strip()
                    candidates.append(
                        {
                            "title": title,
                            "url": url_match.group(1).strip(),
                            "snippet": snippet[:3_000],
                        }
                    )

        hits: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for index, item in enumerate(candidates, 1):
            if not isinstance(item, dict):
                continue
            url = next(
                (
                    str(item[key]).strip()
                    for key in ("url", "link", "html_url", "pdf_url", "entry_id")
                    if item.get(key)
                ),
                "",
            )
            if not url.startswith(("http://", "https://")):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = next(
                (
                    str(item[key]).strip()
                    for key in ("title", "name")
                    if item.get(key)
                ),
                url,
            )
            snippet = next(
                (
                    str(item[key]).strip()
                    for key in ("snippet", "description", "abstract", "summary", "content")
                    if item.get(key)
                ),
                "",
            )
            published_at = next(
                (
                    str(item[key]).strip()
                    for key in (
                        "published_at",
                        "published",
                        "published_date",
                        "publication_date",
                        "date",
                    )
                    if item.get(key)
                ),
                None,
            )
            hits.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "provider": f"mcp:{payload['server']}",
                    "rank": len(hits) + 1,
                    **({"published_at": published_at} if published_at else {}),
                }
            )
        return hits
