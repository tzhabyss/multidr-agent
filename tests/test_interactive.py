from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.interactive import InteractiveShell
from deepresearch.runtime_config import load_runtime_config
from deepresearch.service import ResearchService
from tests.fakes import FakeFetcher, FakeSearchBackend
from tests.test_memory_session import SessionAwareFakeLLM


class CapturedOutput:
    def __init__(self):
        self.values: list[str] = []

    def print(self, value="") -> None:
        self.values.append(str(value))

    @property
    def text(self) -> str:
        return "\n".join(self.values)


def test_cli_parser_accepts_shell_options():
    from deepresearch.cli import _parser

    args = _parser().parse_args(
        [
            "shell",
            "--session-id",
            "session-known",
            "--max-workers",
            "2",
            "--progress",
            "verbose",
            "--heartbeat-seconds",
            "5",
        ]
    )

    assert args.command == "shell"
    assert args.session_id == "session-known"
    assert args.max_workers == 2
    assert args.progress == "verbose"
    assert args.heartbeat_seconds == 5


@pytest.mark.asyncio
async def test_shell_manages_sessions_and_never_sends_commands_to_llm(tmp_path: Path):
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
    llm = SessionAwareFakeLLM()
    output = CapturedOutput()
    async with ResearchService.in_memory(
        llm=llm,
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    ) as service:
        shell = InteractiveShell(service, output=output)
        first = await shell.start()

        assert await shell.handle_line("/new Named conversation") is True
        second_id = shell.session_id
        assert second_id != first.session_id
        assert await shell.handle_line("/sessions") is True
        assert await shell.handle_line(f"/switch {first.session_id}") is True
        assert await shell.handle_line("/does-not-exist") is True
        assert await shell.handle_line("/progress verbose") is True
        assert await shell.handle_line("/mode deep") is True
        assert await shell.handle_line("/mode") is True
        assert await shell.handle_line("/profile drb_evaluation") is True
        assert await shell.handle_line("/mode standard") is True
        assert await shell.handle_line("/profile") is True
        assert await shell.handle_line("/profile missing-profile") is True
        shell.request_overrides["options"] = {"max_workers": 2}
        assert await shell.handle_line("/mode deep") is True
        assert await shell.handle_line("/switch missing-session") is True

        sessions = await service.memory.list_sessions("local-user")
        assert {item.session_id for item in sessions} == {
            first.session_id,
            second_id,
        }
        assert not llm.calls
        assert "Named conversation" in output.text
        assert "Unknown command" in output.text
        assert "Session not found: missing-session" in output.text
        assert "Progress level set to: verbose" in output.text
        assert "Mode set to: deep" in output.text
        assert "mode: deep" in output.text
        assert "Profile set to: drb_evaluation" in output.text
        assert "Mode set to: standard" in output.text
        assert "profile=drb_evaluation | workers=8" in output.text
        assert "profile: drb_evaluation" in output.text
        assert "Unknown profile: missing-profile" in output.text
        assert "Mode set to: deep" in output.text
        assert "profile=drb_evaluation | workers=2" in output.text
        assert await shell.handle_line("/quit") is False


@pytest.mark.asyncio
async def test_shell_runs_chat_and_exposes_history_report_sources_and_memory(tmp_path: Path):
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
    output = CapturedOutput()
    opened: list[Path] = []
    async with ResearchService.in_memory(
        llm=SessionAwareFakeLLM(),
        search_backend=FakeSearchBackend(),
        fetcher=FakeFetcher(),
        artifact_root=tmp_path / "artifacts",
        runtime_config=config,
    ) as service:
        shell = InteractiveShell(
            service,
            output=output,
            report_opener=lambda path: opened.append(path) is None,
        )
        await shell.start()
        await shell.handle_line("What is the verified fact?")
        shell.last_run = None
        await shell.handle_line("/current")
        await shell.handle_line("/history 2")
        await shell.handle_line("/status")
        await shell.handle_line("/report")
        await shell.handle_line("/sources")
        await shell.handle_line("/memory")
        await shell.handle_line("/open")

        session = await service.memory.get_session("local-user", shell.session_id)

    assert session.title == "What is the verified fact?"
    assert len(session.messages) == 2
    assert shell.last_run is not None
    assert "42 units" in output.text
    assert "example.com" in output.text
    assert "rolling_summary" in output.text
    assert "completed_verified" in output.text
    assert opened == [Path(shell.last_run.output_path).resolve()]
