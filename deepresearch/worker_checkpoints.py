"""Durable terminal Worker results inside a LangGraph frontier node.

This journal complements graph checkpoints; it does not resume an in-flight
LLM/tool call or provide exactly-once external side effects.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def window_identity(state: dict[str, Any]) -> str:
    # Bind to the exact input checkpoint, plan, model mode and Skill snapshot.
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class WorkerCheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.serde = JsonPlusSerializer()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("""CREATE TABLE IF NOT EXISTS worker_results (
            run_id TEXT NOT NULL, window_id TEXT NOT NULL, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, payload BLOB NOT NULL, checksum TEXT NOT NULL,
            PRIMARY KEY (run_id, window_id, task_id))""")
        return conn

    def save(self, run_id: str, window_id: str, task_id: str, result: dict) -> None:
        kind, payload = self.serde.dumps_typed(result)
        checksum = hashlib.sha256(kind.encode() + payload).hexdigest()
        conn = self._connect()
        try:
            with conn:
                existing = conn.execute(
                    "SELECT checksum FROM worker_results WHERE run_id=? AND window_id=? AND task_id=?",
                    (run_id, window_id, task_id),
                ).fetchone()
                if existing and existing[0] != checksum:
                    raise ValueError("conflicting Worker checkpoint; concurrent resume is unsupported")
                conn.execute("INSERT OR IGNORE INTO worker_results VALUES (?,?,?,?,?,?)",
                             (run_id, window_id, task_id, kind, payload, checksum))
        finally:
            conn.close()

    def load(self, run_id: str, window_id: str) -> dict[str, dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT task_id, kind, payload, checksum FROM worker_results "
                "WHERE run_id=? AND window_id=? ORDER BY rowid", (run_id, window_id),
            ).fetchall()
        finally:
            conn.close()
        results = {}
        for task_id, kind, payload, checksum in rows:
            if hashlib.sha256(kind.encode() + payload).hexdigest() != checksum:
                raise ValueError("corrupt Worker checkpoint")
            result = self.serde.loads_typed((kind, payload))
            if not isinstance(result, dict):
                raise ValueError("invalid Worker checkpoint result")
            results[task_id] = result
        return results
