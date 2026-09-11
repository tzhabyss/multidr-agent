from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from deepresearch.schemas import ChatResult, ProgressEvent
from deepresearch.web.app import create_app


class _Dumpable:
    def __init__(self, value):
        self.value = value

    def model_dump(self, *, mode: str = "python"):
        del mode
        return dict(self.value)


class _Facts:
    async def stats(self):
        return _Dumpable(
            {
                "active_facts": 3,
                "indexed_documents": 2,
                "embedding_model": "test-embedding",
            }
        )


class _Memory:
    def __init__(self):
        self.facts = _Facts()
        self.local = self
        self.session = SimpleNamespace(
            session_id="session-saved",
            title="Saved research",
            messages=[
                SimpleNamespace(
                    role="user",
                    content="Compare two retrieval approaches.",
                    created_at="2026-09-10T09:00:00+08:00",
                    run_id=None,
                )
            ],
            related_run_ids=["run-saved"],
            updated_at="2026-09-10T09:00:00+08:00",
        )

    async def list_sessions(self, user_id, limit, project_id=None, offset=0):
        assert user_id == "local-user"
        assert limit == 51
        assert project_id is None
        return [self.session] if offset == 0 else []

    async def archived_message_count(self, user_id, project_id, session_id):
        assert user_id == "local-user"
        assert project_id == "deepresearch-default"
        assert session_id == self.session.session_id
        return 1

    async def archived_messages(
        self, user_id, project_id, session_id, limit=200, offset=0
    ):
        assert user_id == "local-user"
        assert project_id == "deepresearch-default"
        assert session_id == self.session.session_id
        if offset:
            return []
        return [
            {
                "message_id": "message-saved",
                "role": "user",
                "content": "Compare two retrieval approaches.",
                "created_at": "2026-09-10T09:00:00+08:00",
                "run_id": None,
            }
        ][:limit]

    async def find_session(self, user_id, session_id):
        assert user_id == "local-user"
        return self.session if session_id == self.session.session_id else None

    async def get_session(self, user_id, session_id):
        assert user_id == "local-user"
        if session_id != self.session.session_id:
            self.session = SimpleNamespace(
                session_id=session_id,
                title=None,
                messages=[],
                related_run_ids=[],
                updated_at="2026-09-10T09:00:00+08:00",
            )
        return self.session

    async def set_session_title(self, session, title):
        session.title = title
        return session


class _SkillRuntime:
    def snapshot(self):
        return {
            "snapshot_id": "snapshot-test",
            "skills": {
                "web-search": {
                    "version": "1.0.0",
                    "description": "Search the web",
                    "roles": ["worker"],
                    "profiles": ["general_research"],
                    "instructions": "must not be exposed",
                }
            },
        }


class _FakeService:
    latest = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.memory = _Memory()
        self.skill_runtime = _SkillRuntime()
        self.chat_call = None
        type(self).latest = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def monitor_summary(self):
        return {"agents": {"worker": {"success_rate": 1.0}}}

    async def chat_stream(self, message, **kwargs):
        self.chat_call = {"message": message, **kwargs}
        yield ProgressEvent(
            event_id="event-1",
            sequence=1,
            elapsed_seconds=0.1,
            event_type="phase",
            phase="planning",
            agent_role="planner",
            status="started",
            message="Planning research",
            run_id="run-test",
        )
        yield ChatResult(
            user_id="local-user",
            project_id="default",
            session_id=kwargs["session_id"],
            turn_id="turn-test",
            action="clarify",
            message="Please clarify the time range.",
        )


def test_web_console_serves_runtime_health_and_sanitized_skills():
    app = create_app(service_factory=_FakeService)

    with TestClient(app) as client:
        index = client.get("/")
        health = client.get("/api/health")
        runtime = client.get("/api/runtime")
        skills = client.get("/api/skills")

    assert index.status_code == 200
    assert "今天想研究什么" in index.text
    assert health.json()["memory"]["active_facts"] == 3
    assert runtime.json()["profiles"]["general_research"]["skill_evolution_mode"] == "observe"
    assert skills.json()["skills"] == [
        {
            "skill_id": "web-search",
            "version": "1.0.0",
            "description": "Search the web",
            "roles": ["worker"],
            "profiles": ["general_research"],
        }
    ]
    assert "must not be exposed" not in skills.text


def test_chat_stream_emits_ready_progress_and_result_events():
    app = create_app(service_factory=_FakeService)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "  Compare two retrieval approaches.  ",
                "session_id": "session-test",
                "mode": "quick",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: ready" in response.text
    assert "event: progress" in response.text
    assert "event: result" in response.text
    event_payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert event_payloads[0] == {"session_id": "session-test"}
    assert event_payloads[1]["phase"] == "planning"
    assert event_payloads[2]["action"] == "clarify"
    assert _FakeService.latest.chat_call["message"] == "Compare two retrieval approaches."
    assert _FakeService.latest.chat_call["request_overrides"]["profile_id"] == "general_research"


def test_web_console_lists_conversations_by_title_and_loads_history():
    app = create_app(service_factory=_FakeService)

    with TestClient(app) as client:
        sessions = client.get("/api/sessions")
        detail = client.get("/api/sessions/session-saved")
        missing = client.get("/api/sessions/session-missing")

    assert sessions.status_code == 200
    assert sessions.json()["has_more"] is False
    assert sessions.json()["sessions"][0] == {
        "session_id": "session-saved",
        "title": "Saved research",
        "message_count": 1,
        "updated_at": "2026-09-10T09:00:00+08:00",
        "last_run_id": "run-saved",
    }
    assert detail.status_code == 200
    assert detail.json()["title"] == "Saved research"
    assert detail.json()["messages"][0]["content"] == "Compare two retrieval approaches."
    assert detail.json()["message_count"] == 1
    assert detail.json()["has_more"] is False
    assert missing.status_code == 404


def test_web_console_keeps_research_replies_in_the_conversation_timeline():
    app = create_app(service_factory=_FakeService)

    with TestClient(app) as client:
        script = client.get("/assets/app.js?v=4")

    assert script.status_code == 200
    assert "message-run-action" in script.text
    assert "查看此报告" in script.text
    assert 'message.role !== "assistant" || !message.run_id' not in script.text


def test_chat_stream_rejects_unsafe_session_id():
    app = create_app(service_factory=_FakeService)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "test", "session_id": "../../escape"},
        )

    assert response.status_code == 422
