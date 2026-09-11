"""Codex-like interactive terminal for persistent research sessions."""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Protocol

from .schemas import ChatResult, ProgressEvent, ResearchMode, RunResult, SessionMemory
from .service import ResearchService, RunNotFoundError
from .time_utils import convert_iso, now_iso


COMMANDS = (
    "/monitor",
    "/skills",
    "/new",
    "/rename",
    "/sessions",
    "/switch",
    "/current",
    "/history",
    "/status",
    "/report",
    "/open",
    "/sources",
    "/memory",
    "/mode",
    "/profile",
    "/progress",
    "/resume",
    "/help",
    "/quit",
    "/exit",
)


HELP_TEXT = """Available commands:
  /monitor              Show tool and stage health (no routing changes)
  /skills [reload|activate|rollback]
                        Inspect skills, preview updates, activate approved rules, or roll back
  /new [name]           Create and switch to a new conversation
  /rename <name>        Rename the current conversation
  /sessions [n]         List recent conversations
  /switch <session_id>  Switch to an existing conversation
  /current              Show the current conversation
  /history [n]          Show the latest messages (default: 10)
  /status               Show the latest research run status
  /report               Display the latest research report
  /open                 Open the latest report in VS Code when available
  /sources              List sources from the latest research run
  /memory               Inspect current session and long-term memory
  /mode [quick|standard|deep|max]
                        Show or change the research mode for new turns
  /profile [profile_id] Show or change the agent profile for new turns
  /progress [level]     Set progress: quiet, normal, or verbose
  /resume <run_id>      Resume or load a checkpointed research run
  /help                 Show this help
  /quit, /exit          Exit the shell

Any input without a leading slash is sent to the current conversation."""


class Output(Protocol):
    def print(self, value: Any = "") -> None: ...


class PlainOutput:
    def print(self, value: Any = "") -> None:
        print(value)

    def markdown(self, value: str) -> None:
        print(value)


def terminal_safe_text(value: Any, encoding: str | None) -> str:
    """Make UI text printable without turning a completed run into an error."""
    text = str(value)
    codec = encoding or "utf-8"
    try:
        return text.encode(codec, errors="replace").decode(codec, errors="replace")
    except LookupError:
        return text


def terminal_supports_unicode_markdown(encoding: str | None) -> bool:
    return (encoding or "").lower().replace("_", "-") in {"utf-8", "utf8"}


class ProgressRenderer:
    LEVELS = {"quiet": 0, "normal": 1, "verbose": 2}

    def __init__(self, output: Output, level: str = "normal") -> None:
        self.output = output
        self.level = self.validate_level(level)

    @classmethod
    def validate_level(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in cls.LEVELS:
            raise ValueError("progress level must be quiet, normal, or verbose")
        return normalized

    def render(self, event: ProgressEvent) -> None:
        if self.LEVELS[event.visibility] > self.LEVELS[self.level]:
            return
        timestamp = event.timestamp
        try:
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
        except ValueError:
            timestamp = timestamp[:8]
        elapsed = int(event.elapsed_seconds)
        label = (event.worker_id or event.agent_role).upper()
        self.output.print(
            f"[{timestamp} | +{elapsed // 60:02d}:{elapsed % 60:02d}] "
            f"{label:<12} {event.message}"
        )
        if event.event_type == "planner_completed":
            for task in event.metrics.get("tasks", []):
                self.output.print(
                    f"  {task.get('worker_id')}  {task.get('dimension')}: "
                    f"{task.get('objective')}"
                )
        if event.event_type == "worker_completed":
            metrics = event.metrics
            total = metrics.get("total_workers")
            if total:
                self.output.print(
                    f"  WORKERS R{event.round_no}: {metrics.get('finished_workers', 0)}/{total} "
                    f"完成，{metrics.get('running_workers', 0)} 运行中，"
                    f"成功 {metrics.get('successful_workers', 0)}，"
                    f"材料不足 {metrics.get('insufficient_workers', 0)}，"
                    f"失败 {metrics.get('failed_workers', 0)}"
                )
        if self.level == "verbose":
            details = []
            for key in (
                "query",
                "url",
                "skill_name",
                "result_count",
                "accepted_claims",
                "stop_reason",
                "providers",
                "input_tokens",
                "output_tokens",
                "http_status",
                "content_chars",
            ):
                value = event.metrics.get(key)
                if value not in (None, "", []):
                    details.append(f"{key}={value}")
            if details:
                self.output.print("  " + "; ".join(details))


class InteractiveShell:
    """Testable command dispatcher; terminal input is kept outside this class."""

    def __init__(
        self,
        service: ResearchService,
        *,
        user_id: str = "local-user",
        session_id: str | None = None,
        request_overrides: dict[str, Any] | None = None,
        output: Output | None = None,
        report_opener: Callable[[Path], bool] | None = None,
        progress_level: str = "normal",
        heartbeat_seconds: float = 15.0,
    ):
        self.service = service
        self.user_id = user_id
        self.session_id = session_id
        self.request_overrides = request_overrides or {}
        self.output = output or PlainOutput()
        self.report_opener = report_opener or self._open_in_vscode
        self.last_run: RunResult | None = None
        self.progress = ProgressRenderer(self.output, progress_level)
        self.heartbeat_seconds = heartbeat_seconds

    async def start(self) -> SessionMemory:
        if self.service.memory is None:
            raise RuntimeError("ResearchService must be open before starting the shell")
        if self.session_id:
            session = await self.service.memory.get_session(
                self.user_id, self.session_id
            )
        else:
            self.session_id = self._new_session_id()
            session = await self.service.memory.get_session(
                self.user_id, self.session_id
            )
        return session

    async def handle_line(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True
        if not self.session_id:
            await self.start()
        if line.startswith("/"):
            return await self._handle_command(line)
        await self._handle_message(line)
        return True

    async def _handle_command(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.output.print(f"Command parse error: {exc}")
            return True
        command = parts[0].lower()
        args = parts[1:]
        handlers = {
            "/monitor": self._monitor,
            "/skills": self._skills,
            "/new": self._new,
            "/rename": self._rename,
            "/sessions": self._sessions,
            "/switch": self._switch,
            "/current": self._current,
            "/history": self._history,
            "/status": self._status,
            "/report": self._report,
            "/open": self._open,
            "/sources": self._sources,
            "/memory": self._memory,
            "/mode": self._mode,
            "/profile": self._profile,
            "/progress": self._progress,
            "/resume": self._resume,
            "/help": self._help,
        }
        if command in {"/quit", "/exit"}:
            return False
        handler = handlers.get(command)
        if handler is None:
            self.output.print(f"Unknown command: {command}. Use /help to list commands.")
            return True
        await handler(args)
        return True

    async def _monitor(self, args: list[str]) -> None:
        self.output.print(json.dumps(self.service.monitor_summary(), ensure_ascii=False, indent=2))

    async def _skills(self, args: list[str]) -> None:
        action = args[0] if args else "list"
        if action in {"reload", "activate"}:
            value = self.service.reload_skills(approved=action == "activate")
        elif action == "rollback":
            try:
                self.service.rollback_skills()
                value = {"rolled_back": True}
            except ValueError as exc:
                value = {"error": str(exc)}
        elif action == "list":
            value = self.service.skill_runtime.registry.loading_snapshot()
        else:
            self.output.print("Usage: /skills [reload|activate|rollback]")
            return
        self.output.print(json.dumps(value, ensure_ascii=False, indent=2))

    async def _handle_message(self, message: str) -> None:
        assert self.service.memory is not None and self.session_id is not None
        session = await self.service.memory.get_session(self.user_id, self.session_id)
        if not session.title:
            title = " ".join(message.split())[:48]
            await self.service.memory.set_session_title(session, title)
        result: ChatResult | None = None
        async for item in self.service.chat_stream(
            message,
            session_id=self.session_id,
            user_id=self.user_id,
            request_overrides=self.request_overrides,
            heartbeat_seconds=self.heartbeat_seconds,
        ):
            if isinstance(item, ProgressEvent):
                self.progress.render(item)
            else:
                result = item
        if result is None:
            raise RuntimeError("chat stream ended without a final result")
        self.last_run = result.run or self.last_run
        label = {
            "clarify": "clarifier",
            "research": "research",
            "respond_from_context": "memory",
        }[result.action]
        self.output.print(f"[{label}]")
        self._print_markdown(result.message)
        if result.run is not None:
            self.output.print(
                f"Run {result.run.run_id}: {result.run.status.value} | "
                f"{len(result.run.sources)} sources"
            )

    async def _new(self, args: list[str]) -> None:
        assert self.service.memory is not None
        self.session_id = self._new_session_id()
        session = await self.service.memory.get_session(self.user_id, self.session_id)
        title = " ".join(args).strip() or None
        if title:
            session = await self.service.memory.set_session_title(session, title)
        self.last_run = None
        self.output.print(
            f"Created session: {session.session_id}"
            + (f" ({session.title})" if session.title else "")
        )

    async def _rename(self, args: list[str]) -> None:
        if not args:
            self.output.print("Usage: /rename <name>")
            return
        session = await self._session()
        assert self.service.memory is not None
        session = await self.service.memory.set_session_title(session, " ".join(args))
        self.output.print(f"Renamed session to: {session.title}")

    async def _sessions(self, args: list[str]) -> None:
        limit = self._positive_int(args, default=20, maximum=100)
        if limit is None:
            return
        assert self.service.memory is not None
        sessions = await self.service.memory.list_sessions(self.user_id, limit)
        if not sessions:
            self.output.print("No sessions found.")
            return
        for session in sessions:
            marker = "*" if session.session_id == self.session_id else " "
            title = session.title or self._derived_title(session) or "untitled"
            self.output.print(
                f"{marker} {session.session_id} | {title} | "
                f"{len(session.messages)} messages | "
                f"{convert_iso(session.updated_at, self.service.runtime_config.timezone)}"
            )

    async def _switch(self, args: list[str]) -> None:
        if len(args) != 1:
            self.output.print("Usage: /switch <session_id>")
            return
        assert self.service.memory is not None
        session = await self.service.memory.find_session(self.user_id, args[0])
        if session is None:
            self.output.print(f"Session not found: {args[0]}")
            return
        self.session_id = session.session_id
        self.last_run = None
        self.output.print(
            f"Switched to: {session.session_id}"
            + (f" ({session.title})" if session.title else "")
        )

    async def _current(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /current")
            return
        session = await self._session()
        self.output.print(f"user: {session.user_id}")
        self.output.print(f"project: {session.project_id}")
        self.output.print(f"session: {session.session_id}")
        self.output.print(f"title: {session.title or self._derived_title(session) or 'untitled'}")
        self.output.print(f"messages: {len(session.messages)}")
        self.output.print(f"runs: {len(session.related_run_ids)}")
        self.output.print(f"mode: {self._current_mode()}")
        self.output.print(f"profile: {self._current_profile()}")

    async def _history(self, args: list[str]) -> None:
        limit = self._positive_int(args, default=10, maximum=100)
        if limit is None:
            return
        session = await self._session()
        if not session.messages:
            self.output.print("This session has no messages yet.")
            return
        for message in session.messages[-limit:]:
            suffix = f" ({message.run_id})" if message.run_id else ""
            self.output.print(f"{message.role}{suffix}: {message.content}")

    async def _status(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /status")
            return
        run = await self._latest_run()
        if run is None:
            self.output.print("No research run is associated with this session.")
            return
        self.output.print(f"run_id: {run.run_id}")
        self.output.print(f"mode: {run.mode.value}")
        self.output.print(f"status: {run.status.value}")
        self.output.print(f"sources: {len(run.sources)}; claims: {len(run.claims)}")
        self.output.print(f"output: {run.output_path or 'not written'}")
        if self.service.model_router is not None:
            routing = self.service.model_router.describe(run.mode.value)
            stages = ("planner", "worker_agent_loop", "synthesizer", "report_writer")
            self.output.print(
                "models: "
                + "; ".join(
                    f"{stage}={routing[stage]['tier']}:{routing[stage]['provider']}/{routing[stage]['model']}"
                    for stage in stages
                )
            )

    async def _report(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /report")
            return
        run = await self._latest_run()
        if run is None:
            self.output.print("No report is associated with this session.")
            return
        self._print_markdown(run.markdown or "The run has no report content.")

    async def _open(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /open")
            return
        run = await self._latest_run()
        if run is None or not run.output_path:
            self.output.print("No report is associated with this session.")
            return
        path = Path(run.output_path).resolve()
        if not path.is_file():
            self.output.print(f"Report file does not exist: {path}")
            return
        if self.report_opener(path):
            self.output.print(f"Opened: {path}")
        else:
            self.output.print(f"VS Code command was not found. Report: {path}")

    async def _sources(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /sources")
            return
        run = await self._latest_run()
        if run is None:
            self.output.print("No sources are associated with this session.")
            return
        if not run.sources:
            self.output.print("The latest run has no sources.")
            return
        for index, source in enumerate(run.sources, 1):
            self.output.print(
                f"{index}. [Tier {source.quality_tier}] {source.title}\n"
                f"   {source.canonical_uri}"
            )

    async def _memory(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /memory")
            return
        session = await self._session()
        assert self.service.memory is not None
        query = session.current_intent or self._derived_title(session) or ""
        episodes = await self.service.memory.relevant_episodes(session, query, 3)
        stats = await self.service.memory.facts.stats(
            self.service.memory.scope(session.user_id, project_id=session.project_id)
        )
        self.output.print(f"rolling_summary: {session.rolling_summary or '(empty)'}")
        self.output.print(f"active_entities: {', '.join(session.active_entities) or '(none)'}")
        self.output.print(f"user_corrections: {len(session.user_corrections)}")
        self.output.print(
            "preferences: "
            + (json.dumps(session.preferences, ensure_ascii=False) if session.preferences else "{}")
        )
        self.output.print(f"related_runs: {len(session.related_run_ids)}")
        self.output.print(f"relevant_long_term_memories: {len(episodes)}")
        self.output.print(
            "autonomous_memory: "
            f"active={stats.active_facts}, inactive={stats.inactive_facts}, "
            f"pending_capture={stats.pending_captures}, pending_index={stats.pending_indexes}"
        )
        if self.service.memory.facts.last_error:
            self.output.print(
                f"memory_extractor: degraded ({self.service.memory.facts.last_error})"
            )
        for item in episodes:
            summary = str(item.get("summary") or item.get("research_brief") or "")
            self.output.print(
                f"- [{item.get('origin')}] "
                f"{item.get('run_id') or item.get('memory_id')}: {summary}"
            )

    async def _progress(self, args: list[str]) -> None:
        if not args:
            self.output.print(f"progress: {self.progress.level}")
            return
        if len(args) != 1:
            self.output.print("Usage: /progress quiet|normal|verbose")
            return
        try:
            self.progress.level = ProgressRenderer.validate_level(args[0])
        except ValueError as exc:
            self.output.print(str(exc))
            return
        self.output.print(f"Progress level set to: {self.progress.level}")

    async def _mode(self, args: list[str]) -> None:
        if not args:
            self.output.print(f"mode: {self._current_mode()}")
            return
        if len(args) != 1:
            self.output.print("Usage: /mode quick|standard|deep|max")
            return
        try:
            mode = ResearchMode(args[0].strip().lower()).value
        except ValueError:
            self.output.print("Usage: /mode quick|standard|deep|max")
            return
        self.request_overrides["mode"] = mode
        profile_id = self._current_profile()
        effective = self.service.runtime_config.effective_mode_config(mode, profile_id)
        workers = self._effective_workers(effective.max_workers)
        if self.service.model_router is not None:
            selected = self.service.model_router.describe(mode)
            self.output.print(
                f"Mode set to: {mode} | "
                f"planner={selected['planner']['tier']} | "
                f"synthesizer={selected['synthesizer']['tier']} | "
                f"profile={profile_id} | workers={workers}"
            )
        else:
            self.output.print(
                f"Mode set to: {mode} | profile={profile_id} | "
                f"workers={workers}"
            )

    async def _profile(self, args: list[str]) -> None:
        if not args:
            self.output.print(f"profile: {self._current_profile()}")
            return
        if len(args) != 1:
            self.output.print("Usage: /profile <profile_id>")
            return
        profile_id = args[0].strip()
        try:
            policy = self.service.runtime_config.profile_config(profile_id)
        except ValueError:
            available = ", ".join(sorted(self.service.runtime_config.agent_profiles))
            self.output.print(
                f"Unknown profile: {profile_id}. Available profiles: {available}"
            )
            return
        self.request_overrides["profile_id"] = profile_id
        mode = self._current_mode()
        effective = self.service.runtime_config.effective_mode_config(mode, profile_id)
        workers = self._effective_workers(effective.max_workers)
        reviewer = "enabled" if policy.semantic_citation_gate == "repair" else "disabled"
        self.output.print(
            f"Profile set to: {profile_id} | mode={mode} | "
            f"workers={workers} | searches={effective.max_total_searches} | "
            f"fetches={effective.max_total_fetches} | reviewer={reviewer}"
        )

    async def _resume(self, args: list[str]) -> None:
        if len(args) != 1:
            self.output.print("Usage: /resume <run_id>")
            return
        try:
            async for item in self.service.resume_stream(
                args[0], heartbeat_seconds=self.heartbeat_seconds
            ):
                if isinstance(item, ProgressEvent):
                    self.progress.render(item)
                else:
                    self.last_run = item
        except RunNotFoundError as exc:
            self.output.print(str(exc))
            return
        if self.last_run is None:
            self.output.print("Resume stream ended without a run result.")
            return
        self.output.print(
            f"Run {self.last_run.run_id}: {self.last_run.status.value}\n"
            f"Report: {self.last_run.output_path or 'not written'}"
        )

    async def _help(self, args: list[str]) -> None:
        if args:
            self.output.print("Usage: /help")
            return
        self.output.print(HELP_TEXT)

    async def _session(self) -> SessionMemory:
        assert self.service.memory is not None and self.session_id is not None
        return await self.service.memory.get_session(self.user_id, self.session_id)

    def _current_mode(self) -> str:
        return str(self.request_overrides.get("mode", self.service.runtime_config.default_mode))

    def _current_profile(self) -> str:
        return str(self.request_overrides.get("profile_id", "general_research"))

    def _effective_workers(self, configured_default: int) -> int:
        options = self.request_overrides.get("options", {})
        if isinstance(options, dict) and options.get("max_workers") is not None:
            return int(options["max_workers"])
        return configured_default

    async def _latest_run(self) -> RunResult | None:
        session = await self._session()
        if self.last_run is not None:
            return self.last_run
        if not session.related_run_ids:
            return None
        try:
            self.last_run = await self.service.inspect(session.related_run_ids[-1])
        except RunNotFoundError:
            return None
        return self.last_run

    def _positive_int(
        self, args: list[str], *, default: int, maximum: int
    ) -> int | None:
        if len(args) > 1:
            self.output.print("Expected at most one numeric argument.")
            return None
        if not args:
            return default
        try:
            value = int(args[0])
        except ValueError:
            self.output.print(f"Not a number: {args[0]}")
            return None
        if value < 1 or value > maximum:
            self.output.print(f"Value must be between 1 and {maximum}.")
            return None
        return value

    def _print_markdown(self, value: str) -> None:
        renderer = getattr(self.output, "markdown", None)
        if callable(renderer):
            renderer(value)
        else:
            self.output.print(value)

    @staticmethod
    def _derived_title(session: SessionMemory) -> str | None:
        for message in session.messages:
            if message.role == "user":
                return " ".join(message.content.split())[:48]
        return None

    @staticmethod
    def _new_session_id() -> str:
        return f"session-{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _open_in_vscode(path: Path) -> bool:
        executable = shutil.which("code")
        if not executable:
            return False
        subprocess.Popen(
            [executable, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True


async def run_terminal_shell(
    service: ResearchService,
    *,
    user_id: str,
    session_id: str | None,
    request_overrides: dict[str, Any],
    progress_level: str = "normal",
    heartbeat_seconds: float = 15.0,
) -> int:
    """Run the real prompt_toolkit/Rich terminal frontend."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    from rich.console import Console
    from rich.markdown import Markdown

    history_path = Path(".runtime") / "shell-history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    # Some Windows/VS Code terminal combinations report color support but do
    # not render ANSI escape sequences, leaving artifacts such as ``?[35m``
    # in the output. Keep the shell transport plain; Markdown layout still
    # works, while prompt_toolkit continues to provide editing/completion.
    console = Console(
        color_system=None,
        force_terminal=False,
        highlight=False,
        markup=False,
    )

    class RichOutput:
        def print(self, value: Any = "") -> None:
            console.print(
                terminal_safe_text(value, getattr(console.file, "encoding", None)),
                markup=False,
                highlight=False,
            )

        def markdown(self, value: str) -> None:
            encoding = getattr(console.file, "encoding", None)
            safe_value = terminal_safe_text(value, encoding)
            if terminal_supports_unicode_markdown(encoding):
                console.print(Markdown(safe_value))
            else:
                # Rich's Markdown renderer introduces characters such as U+2022
                # that legacy GBK Windows consoles cannot encode.
                console.print(safe_value, markup=False, highlight=False)

    shell = InteractiveShell(
        service,
        user_id=user_id,
        session_id=session_id,
        request_overrides=request_overrides,
        output=RichOutput(),
        progress_level=progress_level,
        heartbeat_seconds=heartbeat_seconds,
    )
    session = await shell.start()
    console.print("DeepResearch interactive shell")
    console.print(f"Session: {session.session_id}")
    console.print(
        f"Timezone: {service.runtime_config.timezone} "
        f"({now_iso(service.runtime_config.timezone)[-6:]}) | "
        f"Mode: {shell._current_mode()} | Profile: {shell._current_profile()} | "
        f"Progress: {progress_level} | Type /help for commands."
    )
    console.print("Ctrl+C clears input; Ctrl+D exits.\n")

    # prompt_toolkit requires a real Windows console screen buffer. Keep the
    # shell scriptable for redirected stdin, CI, and editor task runners.
    if not sys.stdin.isatty():
        for raw_line in sys.stdin:
            line = raw_line.rstrip("\r\n")
            console.print(f"deepresearch> {line}")
            try:
                if not await shell.handle_line(line):
                    break
            except Exception as exc:
                console.print(f"{type(exc).__name__}: {exc}")
        console.print("Goodbye.")
        return 0

    prompt = PromptSession(
        history=FileHistory(str(history_path)),
        completer=WordCompleter(COMMANDS, sentence=True),
        complete_while_typing=False,
    )
    with patch_stdout():
        while True:
            try:
                line = await prompt.prompt_async("deepresearch> ")
            except KeyboardInterrupt:
                console.print("^C")
                continue
            except EOFError:
                break
            try:
                if not await shell.handle_line(line):
                    break
            except asyncio.CancelledError:
                console.print("Current operation interrupted.")
            except KeyboardInterrupt:
                console.print("Current operation interrupted.")
            except Exception as exc:
                console.print(f"{type(exc).__name__}: {exc}")
    console.print("Goodbye.")
    return 0
