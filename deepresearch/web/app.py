"""FastAPI surface for the local Deep Research console."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..runtime_config import load_runtime_config
from ..schemas import ChatResult, ProgressEvent, RunResult
from ..service import ResearchService, RunNotFoundError


WEB_ROOT = Path(__file__).resolve().parent / "static"


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=96)
    user_id: str = Field(default="local-user", min_length=1, max_length=96)
    mode: Literal["quick", "standard", "deep", "max"] = "standard"
    profile_id: Literal["general_research", "drb_evaluation"] = "general_research"
    locale: str = Field(default="zh-CN", min_length=2, max_length=32)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in normalized
        ):
            raise ValueError("session_id may contain only letters, digits, '-' and '_'")
        return normalized


def _sse(event: str, value: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"


def _serialize_result(value: ChatResult | RunResult) -> dict[str, Any]:
    return value.model_dump(mode="json")


def create_app(
    *,
    artifact_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    service_factory: Callable[..., ResearchService] = ResearchService,
) -> FastAPI:
    """Create one local-only console app with a shared runtime service."""

    runtime_config = load_runtime_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = service_factory(
            artifact_root=artifact_root,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        app.state.research_service = await service.__aenter__()
        try:
            yield
        finally:
            await service.__aexit__(None, None, None)

    app = FastAPI(
        title="Deep Research Console",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    def service_from(request: Request) -> ResearchService:
        return request.app.state.research_service

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        service = service_from(request)
        memory = None
        if service.memory is not None:
            memory = (await service.memory.facts.stats()).model_dump(mode="json")
        return {
            "status": "ready",
            "memory": memory,
            "monitor": service.monitor_summary(),
        }

    @app.get("/api/runtime")
    async def runtime() -> dict[str, Any]:
        return {
            "modes": list(runtime_config.research_modes),
            "profiles": {
                profile_id: {
                    "reviewer_mode": profile.reviewer_mode,
                    "requirement_matrix": profile.requirement_matrix,
                    "skill_evolution_mode": runtime_config.effective_skill_evolution_mode(
                        profile_id
                    ),
                }
                for profile_id, profile in runtime_config.agent_profiles.items()
            },
            "default_mode": runtime_config.default_mode,
            "memory_enabled": runtime_config.memory.enabled,
            "embedding_model": runtime_config.memory.embedding_model,
        }

    @app.get("/api/skills")
    async def skills(request: Request) -> dict[str, Any]:
        service = service_from(request)
        snapshot = service.skill_runtime.snapshot()
        return {
            "snapshot_id": snapshot["snapshot_id"],
            "skills": [
                {
                    "skill_id": skill_id,
                    "version": row["version"],
                    "description": row["description"],
                    "roles": row["roles"],
                    "profiles": row["profiles"],
                }
                for skill_id, row in sorted(snapshot["skills"].items())
            ],
        }

    @app.get("/api/sessions")
    async def sessions(
        request: Request,
        user_id: str = "local-user",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        service = service_from(request)
        if service.memory is None:
            raise HTTPException(status_code=503, detail="Memory service is unavailable")
        page_limit = min(max(limit, 1), 100)
        page_offset = max(offset, 0)
        values = await service.memory.list_sessions(
            user_id,
            page_limit + 1,
            offset=page_offset,
        )
        has_more = len(values) > page_limit
        values = values[:page_limit]

        async def message_count(session: Any) -> int:
            archive = getattr(service.memory, "local", None)
            counter = getattr(archive, "archived_message_count", None)
            if counter is None:
                return len(session.messages)
            project_id = getattr(
                session,
                "project_id",
                getattr(getattr(service.memory, "config", None), "project_id", "deepresearch-default"),
            )
            count = await counter(user_id, project_id, session.session_id)
            return count or len(session.messages)

        message_counts = await asyncio.gather(*(message_count(session) for session in values))

        def derived_title(session: Any) -> str:
            if session.title:
                return session.title
            for message in session.messages:
                if message.role == "user":
                    return " ".join(message.content.split())[:48]
            return "新研究"

        return {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "title": derived_title(session),
                    "message_count": message_counts[index],
                    "updated_at": session.updated_at,
                    "last_run_id": (
                        session.related_run_ids[-1]
                        if session.related_run_ids
                        else None
                    ),
                }
                for index, session in enumerate(values)
            ],
            "has_more": has_more,
            "next_offset": page_offset + len(values) if has_more else None,
        }

    @app.get("/api/sessions/{session_id}")
    async def session(
        session_id: str,
        request: Request,
        user_id: str = "local-user",
        limit: int = 80,
        offset: int = 0,
    ) -> dict[str, Any]:
        service = service_from(request)
        if service.memory is None:
            raise HTTPException(status_code=503, detail="Memory service is unavailable")
        value = await service.memory.find_session(user_id, session_id)
        if value is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        title = value.title
        if not title:
            title = next(
                (
                    " ".join(message.content.split())[:48]
                    for message in value.messages
                    if message.role == "user"
                ),
                "新研究",
            )
        page_limit = min(max(limit, 1), 200)
        page_offset = max(offset, 0)
        archive = getattr(service.memory, "local", None)
        archive_reader = getattr(archive, "archived_messages", None)
        archive_counter = getattr(archive, "archived_message_count", None)
        archived: list[dict[str, Any]] = []
        project_id = getattr(
            value,
            "project_id",
            getattr(getattr(service.memory, "config", None), "project_id", "deepresearch-default"),
        )
        if archive_reader is not None:
            archived = await archive_reader(
                user_id,
                project_id,
                session_id,
                page_limit + 1,
                page_offset,
            )
        total_message_count = (
            await archive_counter(user_id, project_id, session_id)
            if archive_counter is not None
            else len(value.messages)
        )

        has_more = len(archived) > page_limit
        if has_more:
            # Archive rows are returned in chronological order; the extra row is
            # the oldest one and only signals that another page is available.
            archived = archived[1:]

        hot_by_id = {
            getattr(message, "message_id", ""): message for message in value.messages
        }

        def archived_content(row: dict[str, Any]) -> str:
            hot = hot_by_id.get(str(row.get("message_id") or ""))
            if hot is not None:
                return hot.content
            content = str(row.get("content") or "")
            if row.get("run_id") and len(content) > 3_000:
                return content[:3_000].rstrip() + "\n\n完整内容请点击“查看此报告”。"
            return content

        if archived:
            messages = [
                {
                    "message_id": row.get("message_id"),
                    "role": row.get("role"),
                    "content": archived_content(row),
                    "created_at": row.get("created_at"),
                    "run_id": row.get("run_id"),
                }
                for row in archived
                if row.get("role") in {"user", "assistant"}
            ]
        else:
            messages = [
                {
                    "message_id": getattr(message, "message_id", None),
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                    "run_id": message.run_id,
                }
                for message in value.messages
                if message.role in {"user", "assistant"}
            ]

        return {
            "session_id": value.session_id,
            "title": title,
            "messages": messages,
            "related_run_ids": value.related_run_ids,
            "updated_at": value.updated_at,
            "message_count": total_message_count or len(value.messages),
            "has_more": has_more,
            "next_offset": page_offset + len(messages) if has_more else None,
        }

    async def chat_events(
        service: ResearchService,
        payload: ChatStreamRequest,
    ) -> AsyncIterator[str]:
        session_id = payload.session_id or f"session-{uuid.uuid4().hex[:16]}"
        yield _sse("ready", {"session_id": session_id})
        try:
            if service.memory is not None:
                session = await service.memory.get_session(payload.user_id, session_id)
                if not session.title:
                    await service.memory.set_session_title(
                        session,
                        " ".join(payload.message.split())[:48],
                    )
            async for item in service.chat_stream(
                payload.message,
                session_id=session_id,
                user_id=payload.user_id,
                request_overrides={
                    "mode": payload.mode,
                    "profile_id": payload.profile_id,
                    "constraints": {"locale": payload.locale},
                },
                heartbeat_seconds=10.0,
            ):
                if isinstance(item, ProgressEvent):
                    yield _sse("progress", item.model_dump(mode="json"))
                else:
                    yield _sse("result", _serialize_result(item))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield _sse(
                "error",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "session_id": session_id,
                },
            )

    @app.post("/api/chat/stream")
    async def chat_stream(
        payload: ChatStreamRequest,
        request: Request,
    ) -> StreamingResponse:
        return StreamingResponse(
            chat_events(service_from(request), payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    async def resume_events(
        service: ResearchService,
        run_id: str,
    ) -> AsyncIterator[str]:
        try:
            async for item in service.resume_stream(run_id, heartbeat_seconds=10.0):
                if isinstance(item, ProgressEvent):
                    yield _sse("progress", item.model_dump(mode="json"))
                else:
                    yield _sse("result", _serialize_result(item))
        except RunNotFoundError as exc:
            yield _sse("error", {"error_type": "RunNotFoundError", "message": str(exc)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield _sse("error", {"error_type": type(exc).__name__, "message": str(exc)})

    @app.post("/api/runs/{run_id}/resume/stream")
    async def resume_stream(run_id: str, request: Request) -> StreamingResponse:
        return StreamingResponse(
            resume_events(service_from(request), run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}")
    async def inspect_run(run_id: str, request: Request) -> dict[str, Any]:
        try:
            return _serialize_result(await service_from(request).inspect(run_id))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
