"""Bounded tool execution and request-local telemetry; never changes routing."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_ttl_seconds: float = Field(default=0, ge=0)
    cache_read_only: bool = False
    failure_threshold: int = Field(default=3, ge=1)
    recovery_seconds: float = Field(default=120, gt=0)
    circuit_enabled: bool = False


class GovernanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    monitor_interval_seconds: float = Field(default=10, gt=0)
    max_records: int = Field(default=5000, ge=1)
    max_cache_entries: int = Field(default=128, ge=0)
    policies: dict[str, ToolPolicy] = Field(default_factory=dict)
    rewrite_enabled: bool = True
    rerank_enabled: bool = True


_context: ContextVar[dict] = ContextVar("operation_context", default={})
_parent: ContextVar[str | None] = ContextVar("operation_parent", default=None)
_runtime: ContextVar[Any] = ContextVar("operation_runtime", default=None)


@contextmanager
def operation_context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def bind_runtime(runtime):
    token = _runtime.set(runtime)
    try:
        yield
    finally:
        _runtime.reset(token)


def current_runtime():
    return _runtime.get()


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class Circuit:
    failures: int = 0
    opened_until: float = 0
    probe: bool = False

    def state(self):
        return "half_open" if self.probe else "open" if self.opened_until else "closed"


def error_category(exc):
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, CircuitOpenError):
        return "circuit_open"
    if isinstance(exc, PermissionError) or "Authorization" in type(exc).__name__:
        return "authorization"
    if isinstance(exc, (ValueError, TypeError)):
        return "validation"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if exc.__cause__ is not None:
        return error_category(exc.__cause__)
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limit"
    if isinstance(status, int):
        return "service" if status >= 500 else "request"
    name = type(exc).__name__.lower()
    if isinstance(exc, ConnectionError) or any(x in name for x in ("connect", "transport")):
        return "connection"
    return "unknown"


class ToolRuntime:
    def __init__(self, config=None):
        self.config = config or GovernanceConfig()
        self.records = deque(maxlen=self.config.max_records)
        self.circuits: dict[tuple, Circuit] = {}
        self.cache = OrderedDict()
        self.run_records: dict[str, list] = {}
        self.dropped: dict[str, int] = {}

    def record(self, row):
        self.records.append(row)
        run_id = row.get("run_id")
        if run_id in self.run_records:
            rows = self.run_records[run_id]
            if len(rows) < self.config.max_records:
                rows.append(row)
            else:
                self.dropped[run_id] = self.dropped.get(run_id, 0) + 1

    async def execute(self, name, operation, *, arguments=None, timeout=None,
                      kind="tool", version="", identity="", preflight=None, **metadata):
        if not self.config.enabled:
            if preflight:
                preflight()
            return await operation()
        policy = self.config.policies.get(name, ToolPolicy())
        ctx = dict(_context.get())
        op_id = uuid.uuid4().hex
        row = {**ctx, **metadata, "operation_id": op_id,
               "parent_operation_id": _parent.get(), "name": name,
               "kind": kind, "version": version, "status": "failed",
               "cache_status": "miss", "error_category": None,
               "started_at": time.time()}
        parent_token = _parent.set(op_id)
        start = time.monotonic()
        # Scope health to a run: evaluation runs cannot contaminate each other.
        key = (ctx.get("run_id", "local"), name, version, identity)
        circuit = self.circuits.setdefault(key, Circuit())
        admitted = False
        cache_key = None
        try:
            if preflight:
                preflight()
            if policy.cache_read_only and policy.cache_ttl_seconds and ctx.get("run_id"):
                digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()
                cache_key = (*key, ctx.get("profile_id"), digest)
                cached = self.cache.get(cache_key)
                if cached and cached[0] > time.monotonic():
                    self.cache.move_to_end(cache_key)
                    row.update(status="success", cache_status="hit")
                    return copy.deepcopy(cached[1])
                self.cache.pop(cache_key, None)
            if kind == "tool" and policy.circuit_enabled:
                if circuit.probe or circuit.opened_until > time.monotonic():
                    raise CircuitOpenError(name)
                if circuit.opened_until:
                    circuit.probe = True
                admitted = True
            call = operation()
            value = await asyncio.wait_for(call, timeout) if timeout else await call
            row["status"] = "empty" if value is None or value == [] else "success"
            if admitted:
                circuit.failures = 0
                circuit.opened_until = 0
            if cache_key and row["status"] == "success" and self.config.max_cache_entries:
                self.cache[cache_key] = (time.monotonic() + policy.cache_ttl_seconds, copy.deepcopy(value))
                while len(self.cache) > self.config.max_cache_entries:
                    self.cache.popitem(last=False)
            return value
        except BaseException as exc:
            category = error_category(exc)
            row["error_category"] = category
            row["status"] = "cancelled" if category == "cancelled" else "failed"
            # Unknown, user, policy and empty-result failures never trip transport health.
            if admitted and category in {"timeout", "connection", "rate_limit", "service"}:
                circuit.failures += 1
                if circuit.probe or circuit.failures >= policy.failure_threshold:
                    circuit.opened_until = time.monotonic() + policy.recovery_seconds
            raise
        finally:
            if admitted:
                circuit.probe = False
            row.update(latency_ms=(time.monotonic() - start) * 1000,
                       circuit_state=circuit.state(), finished_at=time.time())
            self.record(row)
            _parent.reset(parent_token)

    def release_run(self, run_id):
        self.run_records.pop(run_id, None)
        self.dropped.pop(run_id, None)
        for mapping in (self.cache, self.circuits):
            for key in list(mapping):
                if key[0] == run_id:
                    del mapping[key]
