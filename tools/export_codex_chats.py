#!/usr/bin/env python3
"""Export user-visible Codex chat messages for one local project.

The exporter intentionally omits system/developer prompts, reasoning, tool calls,
tool outputs, permission snapshots, and other machine-specific runtime state.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INJECTED_PREFIXES = (
    "<environment_context>",
    "<recommended_plugins>",
    "<permissions instructions>",
    "<turn_aborted>",
)

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{20,}"),
)

ROLLOUT_ID_PATTERN = re.compile(
    r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", re.IGNORECASE
)


@dataclass
class Message:
    role: str
    text: str
    timestamp: str
    phase: str | None = None


@dataclass
class Chat:
    thread_id: str
    cwd: str
    created_at: str
    updated_at: str
    archived: bool
    source_file: Path
    title: str
    messages: list[Message]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalized_path(value: str | Path) -> str:
    return str(Path(value).resolve()).rstrip("\\/").casefold()


def iter_rollouts(codex_home: Path) -> Iterable[tuple[Path, bool]]:
    sessions = codex_home / "sessions"
    archived = codex_home / "archived_sessions"
    if sessions.exists():
        for path in sessions.rglob("*.jsonl"):
            yield path, False
    if archived.exists():
        for path in archived.rglob("*.jsonl"):
            yield path, True


def extract_text(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    values: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {"input_text", "output_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            values.append(text.strip())
    return values


def is_injected_context(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in INJECTED_PREFIXES)


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)(Bearer"):
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def title_from_messages(messages: list[Message], thread_id: str) -> str:
    for message in messages:
        if message.role != "user":
            continue
        title = re.sub(r"\s+", " ", message.text).strip()
        if title:
            return title[:80] + ("…" if len(title) > 80 else "")
    return f"Codex chat {thread_id}"


def rollout_id_from_path(path: Path) -> str:
    matches = ROLLOUT_ID_PATTERN.findall(path.name)
    return matches[-1] if matches else path.stem


def read_chat(path: Path, archived: bool, project_key: str) -> tuple[Chat | None, str | None]:
    meta: dict[str, Any] | None = None
    messages: list[Message] = []
    first_timestamp = ""
    last_timestamp = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = str(record.get("timestamp") or "")
            if timestamp:
                first_timestamp = first_timestamp or timestamp
                last_timestamp = timestamp

            if record.get("type") == "session_meta":
                payload = record.get("payload")
                if isinstance(payload, dict):
                    meta = payload
                continue

            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue

            parts = []
            for text in extract_text(payload.get("content")):
                if role == "user" and is_injected_context(text):
                    continue
                parts.append(redact_secrets(text))
            visible = "\n\n".join(parts).strip()
            if not visible:
                continue
            messages.append(
                Message(
                    role=role,
                    text=visible,
                    timestamp=timestamp,
                    phase=payload.get("phase") if role == "assistant" else None,
                )
            )

    if not meta:
        return None, "missing_session_metadata"
    cwd = str(meta.get("cwd") or "")
    if not cwd or normalized_path(cwd) != project_key:
        return None, "different_working_directory"

    thread_id = str(meta.get("session_id") or meta.get("id") or path.stem)
    # Child/evaluator rollouts can inherit a parent session_id while using a
    # different rollout UUID. They are internal runs, not sidebar chats.
    if rollout_id_from_path(path).casefold() != thread_id.casefold():
        return None, "noncanonical_child_rollout"
    created_at = str(meta.get("timestamp") or first_timestamp)
    return Chat(
        thread_id=thread_id,
        cwd=cwd,
        created_at=created_at,
        updated_at=last_timestamp or created_at,
        archived=archived,
        source_file=path,
        title=title_from_messages(messages, thread_id),
        messages=messages,
    ), None


def safe_filename(chat: Chat) -> str:
    date = chat.created_at[:10] if chat.created_at else "unknown-date"
    title = re.sub(r"[<>:\"/\\|?*#\x00-\x1f]", "-", chat.title)
    title = re.sub(r"\s+", " ", title).strip(" .-")[:70] or "chat"
    return f"{date}--{title}--{chat.thread_id}.md"


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_chat(chat: Chat, codex_home: Path) -> str:
    try:
        source = chat.source_file.relative_to(codex_home)
    except ValueError:
        source = chat.source_file

    lines = [
        "---",
        f"title: {yaml_string(chat.title)}",
        f"thread_id: {yaml_string(chat.thread_id)}",
        f"created_at: {yaml_string(chat.created_at)}",
        f"updated_at: {yaml_string(chat.updated_at)}",
        f"original_cwd: {yaml_string(chat.cwd)}",
        f"archived: {'true' if chat.archived else 'false'}",
        f"message_count: {len(chat.messages)}",
        f"source_rollout: {yaml_string(str(source))}",
        "export_scope: user_and_assistant_visible_messages",
        "secrets_redacted: true",
        "---",
        "",
        f"# {chat.title}",
        "",
        "> Portable transcript exported from a local Codex task. System prompts, reasoning, tool calls, tool outputs, and runtime state are omitted.",
        "",
    ]

    for message in chat.messages:
        if message.role == "user":
            heading = "## User"
        else:
            phase = f" · {message.phase}" if message.phase else ""
            heading = f"## Assistant{phase}"
        if message.timestamp:
            heading += f" · {message.timestamp}"
        lines.extend([heading, "", message.text, ""])

    return "\n".join(lines).rstrip() + "\n"


def render_index(chats: list[Chat], filenames: dict[str, str], exported_at: str) -> str:
    lines = [
        "# Codex chat archive",
        "",
        f"Exported at: `{exported_at}`",
        "",
        "This archive contains user-visible messages from local Codex tasks whose recorded working directory exactly matched this project. It excludes system/developer prompts, hidden reasoning, tool calls, tool outputs, and runtime configuration. Obvious credential patterns are redacted.",
        "",
        "On macOS, open this repository as a local project and ask Codex to read this index plus the relevant transcript before continuing the work.",
        "",
        "Suggested prompt:",
        "",
        "```text",
        "Read docs/codex-chat-archive/README.md and the relevant transcript files. Restore the prior project context, inspect the current repository state, and continue from the latest unfinished decision.",
        "```",
        "",
        f"## Chats ({len(chats)})",
        "",
        "| Created | Status | Messages | Title | Thread ID |",
        "|---|---|---:|---|---|",
    ]
    for chat in chats:
        status = "archived" if chat.archived else "active/recent"
        filename = filenames[chat.thread_id]
        title = chat.title.replace("|", "\\|")
        lines.append(
            f"| {chat.created_at[:10] or '-'} | {status} | {len(chat.messages)} | [{title}](<{filename}>) | `{chat.thread_id}` |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "These Markdown files are a portable archive, not an importable Codex task database. Continuing on another computer creates a new task with the archived transcript as context.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_key = normalized_path(args.project_dir)
    chats: list[Chat] = []
    skipped_noncanonical = 0
    for rollout, archived in iter_rollouts(args.codex_home):
        chat, skip_reason = read_chat(rollout, archived, project_key)
        if skip_reason == "noncanonical_child_rollout":
            skipped_noncanonical += 1
        if chat is not None:
            chats.append(chat)

    chats.sort(key=lambda chat: (chat.created_at, chat.thread_id), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_manifest_path = args.output_dir / "manifest.json"
    if old_manifest_path.exists():
        try:
            old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
            for item in old_manifest.get("chats", []):
                old_name = item.get("file")
                if not isinstance(old_name, str) or Path(old_name).name != old_name:
                    continue
                old_path = args.output_dir / old_name
                if old_path.is_file():
                    old_path.unlink()
        except (json.JSONDecodeError, OSError):
            pass

    filenames: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    for chat in chats:
        filename = safe_filename(chat)
        filenames[chat.thread_id] = filename
        (args.output_dir / filename).write_text(
            render_chat(chat, args.codex_home), encoding="utf-8", newline="\n"
        )
        manifest.append(
            {
                "thread_id": chat.thread_id,
                "title": chat.title,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
                "archived": chat.archived,
                "message_count": len(chat.messages),
                "file": filename,
            }
        )

    exported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (args.output_dir / "README.md").write_text(
        render_index(chats, filenames, exported_at), encoding="utf-8", newline="\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "exported_at": exported_at,
                "project_dir": str(args.project_dir.resolve()),
                "scope": "user_and_assistant_visible_messages",
                "secrets_redacted": True,
                "chat_count": len(chats),
                "skipped_noncanonical_child_rollouts": skipped_noncanonical,
                "chats": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Exported {len(chats)} chats to {args.output_dir} "
        f"(skipped {skipped_noncanonical} internal child rollouts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
