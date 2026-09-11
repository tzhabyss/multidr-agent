"""Deterministic task allocation and shared search/fetch resource leases."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .ids import make_id
from .schemas import RunBudget, TaskBudget, UsageEvent, utc_now


@dataclass
class _FetchRound:
    total: int
    reservations: dict[str, int]
    reclaimed: int = 0
    consumed: int = 0
    completed: set[str] = field(default_factory=set)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class SharedFetchBudgetPool:
    """Lease one round's fetch budget across concurrent workers.

    Workers consume their initial reservation first. A worker that finishes
    early returns its unused reservation to a shared queue, where another
    worker can borrow it without increasing the round or run budget.
    """

    def __init__(self) -> None:
        self._rounds: dict[tuple[str, int], _FetchRound] = {}
        self._guard = asyncio.Lock()

    async def configure_round(
        self,
        run_id: str,
        round_no: int,
        allocations: dict[str, int],
    ) -> None:
        reservations = {
            task_id: max(0, int(value)) for task_id, value in allocations.items()
        }
        key = (run_id, round_no)
        async with self._guard:
            self._rounds[key] = _FetchRound(
                total=sum(reservations.values()),
                reservations=reservations,
            )

    async def acquire(
        self,
        run_id: str,
        round_no: int,
        task_id: str,
        *,
        wait_timeout_seconds: float = 30.0,
    ) -> bool | None:
        """Return True for a lease, False when the pool is spent, or None if absent."""

        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return None

        async with round_pool.condition:
            while True:
                own = round_pool.reservations.get(task_id, 0)
                if own > 0:
                    round_pool.reservations[task_id] = own - 1
                    round_pool.consumed += 1
                    return True
                if round_pool.reclaimed > 0:
                    round_pool.reclaimed -= 1
                    round_pool.consumed += 1
                    return True
                # Reservations are a fairness preference, not a hard partition.
                # Borrow one unused lease from another active worker so the
                # round behaves as a genuinely shared queue. Completion still
                # reclaims whatever remains.
                donor = next(
                    (
                        owner
                        for owner, remaining in sorted(
                            round_pool.reservations.items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                        if owner != task_id
                        and owner not in round_pool.completed
                        and remaining > 0
                    ),
                    None,
                )
                if donor is not None:
                    round_pool.reservations[donor] -= 1
                    round_pool.consumed += 1
                    return True
                held_by_running_workers = any(
                    owner not in round_pool.completed and remaining > 0
                    for owner, remaining in round_pool.reservations.items()
                    if owner != task_id
                )
                if not held_by_running_workers:
                    return False
                try:
                    await asyncio.wait_for(
                        round_pool.condition.wait(), timeout=wait_timeout_seconds
                    )
                except TimeoutError:
                    return False

    async def complete(self, run_id: str, round_no: int, task_id: str) -> int:
        """Return a completed worker's unused reservation to the queue."""

        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return 0
        async with round_pool.condition:
            if task_id in round_pool.completed:
                return 0
            released = round_pool.reservations.get(task_id, 0)
            round_pool.reservations[task_id] = 0
            round_pool.reclaimed += released
            round_pool.completed.add(task_id)
            round_pool.condition.notify_all()
            return released

    async def release(self, run_id: str, round_no: int, task_id: str) -> bool:
        """Return an unused acquired lease, for example after URL coalescing."""

        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return False
        async with round_pool.condition:
            if round_pool.consumed <= 0:
                return False
            if task_id in round_pool.completed:
                round_pool.reclaimed += 1
            else:
                round_pool.reservations[task_id] = (
                    round_pool.reservations.get(task_id, 0) + 1
                )
            round_pool.consumed -= 1
            round_pool.condition.notify_all()
            return True

    async def snapshot(self, run_id: str, round_no: int) -> dict[str, object] | None:
        """Expose deterministic pool state for audit tests and diagnostics."""

        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return None
        async with round_pool.condition:
            return {
                "total": round_pool.total,
                "consumed": round_pool.consumed,
                "reclaimed": round_pool.reclaimed,
                "reservations": dict(round_pool.reservations),
                "completed": sorted(round_pool.completed),
            }


@dataclass
class _SearchRound:
    total: int
    reservations: dict[str, int]
    consumed: int = 0
    reclaimed: int = 0
    completed: set[str] = field(default_factory=set)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class SharedSearchBudgetPool:
    """Atomically lease logical-query capacity across concurrent workers.

    One lease covers one ``web_search(query)`` call. Provider fanout is an
    implementation detail of the selected search skill and does not multiply
    the user-visible query budget. Atomic leasing still prevents concurrent
    workers from observing stale remaining capacity and overshooting the cap.
    """

    def __init__(self) -> None:
        self._rounds: dict[tuple[str, int], _SearchRound] = {}
        self._guard = asyncio.Lock()

    async def configure_round(
        self,
        run_id: str,
        round_no: int,
        allocations: dict[str, int],
    ) -> None:
        reservations = {
            task_id: max(0, int(value)) for task_id, value in allocations.items()
        }
        async with self._guard:
            self._rounds[(run_id, round_no)] = _SearchRound(
                total=sum(reservations.values()),
                reservations=reservations,
            )

    async def acquire(
        self,
        run_id: str,
        round_no: int,
        task_id: str,
        *,
        units: int = 1,
    ) -> bool | None:
        """Acquire logical-query ``units`` before search execution."""

        units = max(1, int(units))
        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return None

        async with round_pool.condition:
            available = round_pool.reclaimed + sum(
                max(0, value) for value in round_pool.reservations.values()
            )
            if available < units:
                return False

            remaining = units
            own = round_pool.reservations.get(task_id, 0)
            take = min(own, remaining)
            round_pool.reservations[task_id] = own - take
            remaining -= take

            take = min(round_pool.reclaimed, remaining)
            round_pool.reclaimed -= take
            remaining -= take

            if remaining:
                donors = sorted(
                    (
                        (owner, value)
                        for owner, value in round_pool.reservations.items()
                        if owner != task_id and value > 0
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                for owner, value in donors:
                    take = min(value, remaining)
                    round_pool.reservations[owner] = value - take
                    remaining -= take
                    if remaining == 0:
                        break

            if remaining:
                raise RuntimeError("Shared search budget accounting underflow")
            round_pool.consumed += units
            return True

    async def release(
        self,
        run_id: str,
        round_no: int,
        task_id: str,
        *,
        units: int = 1,
    ) -> int:
        """Return logical-query units reserved but not executed."""

        units = max(1, int(units))
        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return 0
        async with round_pool.condition:
            released = min(units, round_pool.consumed)
            if released == 0:
                return 0
            if task_id in round_pool.completed:
                round_pool.reclaimed += released
            else:
                round_pool.reservations[task_id] = (
                    round_pool.reservations.get(task_id, 0) + released
                )
            round_pool.consumed -= released
            round_pool.condition.notify_all()
            return released

    async def complete(self, run_id: str, round_no: int, task_id: str) -> int:
        """Reclaim a completed worker's unused reservation."""

        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return 0
        async with round_pool.condition:
            if task_id in round_pool.completed:
                return 0
            released = round_pool.reservations.get(task_id, 0)
            round_pool.reservations[task_id] = 0
            round_pool.reclaimed += released
            round_pool.completed.add(task_id)
            round_pool.condition.notify_all()
            return released

    async def snapshot(self, run_id: str, round_no: int) -> dict[str, object] | None:
        async with self._guard:
            round_pool = self._rounds.get((run_id, round_no))
        if round_pool is None:
            return None
        async with round_pool.condition:
            return {
                "total": round_pool.total,
                "consumed": round_pool.consumed,
                "reclaimed": round_pool.reclaimed,
                "reservations": dict(round_pool.reservations),
                "completed": sorted(round_pool.completed),
            }


@dataclass
class _DomainHealth:
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    open_domains: set[str] = field(default_factory=set)
    open_until: dict[str, float] = field(default_factory=dict)


class SharedDomainCircuitBreaker:
    """Stop repeatedly spending one run's fetch budget on an unavailable host."""

    def __init__(
        self, *, failure_threshold: int = 2, cooldown_seconds: float = 300.0
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self._runs: dict[str, _DomainHealth] = {}
        self._guard = asyncio.Lock()

    async def can_attempt(self, run_id: str, domain: str) -> bool:
        if not domain:
            return True
        async with self._guard:
            health = self._runs.get(run_id)
            if health is None or domain not in health.open_domains:
                return True
            if health.open_until.get(domain, 0.0) > time.monotonic():
                return False
            # Half-open after the cooldown: one subsequent request may test the
            # host, while a new failure immediately opens it again.
            health.open_domains.discard(domain)
            health.open_until.pop(domain, None)
            health.consecutive_failures[domain] = max(
                0, self.failure_threshold - 1
            )
            return True

    async def record_failure(self, run_id: str, domain: str) -> bool:
        """Record a consecutive failure and return whether the circuit is open."""

        if not domain:
            return False
        async with self._guard:
            health = self._runs.setdefault(run_id, _DomainHealth())
            failures = health.consecutive_failures.get(domain, 0) + 1
            health.consecutive_failures[domain] = failures
            if failures >= self.failure_threshold:
                health.open_domains.add(domain)
                health.open_until[domain] = (
                    time.monotonic() + self.cooldown_seconds
                )
            return domain in health.open_domains

    async def record_success(self, run_id: str, domain: str) -> None:
        if not domain:
            return
        async with self._guard:
            health = self._runs.setdefault(run_id, _DomainHealth())
            health.consecutive_failures.pop(domain, None)
            health.open_domains.discard(domain)
            health.open_until.pop(domain, None)

    async def clear_run(self, run_id: str) -> None:
        async with self._guard:
            self._runs.pop(run_id, None)

    async def snapshot(self, run_id: str) -> dict[str, object]:
        async with self._guard:
            health = self._runs.get(run_id)
            if health is None:
                return {"consecutive_failures": {}, "open_domains": []}
            return {
                "consecutive_failures": dict(health.consecutive_failures),
                "open_domains": sorted(health.open_domains),
            }


@dataclass
class _FetchRunCache:
    completed: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    completed_chars: int = 0
    in_flight: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    url_owners: dict[str, str] = field(default_factory=dict)
    failures: OrderedDict[str, "_NegativeFetchEntry"] = field(
        default_factory=OrderedDict
    )


@dataclass(frozen=True)
class _NegativeFetchEntry:
    error_type: str
    message: str
    expires_at: float


class CachedFetchFailure(RuntimeError):
    """A same-run URL failed recently and is temporarily suppressed."""

    def __init__(self, url_key: str, error_type: str, message: str) -> None:
        super().__init__(f"negative_cache:{error_type}: {message}")
        self.url_key = url_key
        self.error_type = error_type
        self.cached_message = message


class SharedFetchCoordinator:
    """Queue, bound, coalesce, and cache same-run URL fetches.

    Only fetched documents are shared. Evidence extraction and model context
    remain private to each worker. The cache is bounded by entries and content
    characters, while one shared semaphore bounds network concurrency across
    every worker using this coordinator.
    """

    def __init__(
        self,
        *,
        max_entries_per_run: int = 256,
        max_content_chars_per_run: int = 2_000_000,
        max_cached_document_chars: int = 250_000,
        max_concurrency: int = 16,
        max_concurrency_per_domain: int = 4,
        failure_ttl_seconds: float = 120.0,
        deterministic_failure_ttl_seconds: float = 3600.0,
        max_failures_per_run: int = 128,
    ) -> None:
        self.max_entries_per_run = max(1, max_entries_per_run)
        self.max_content_chars_per_run = max(1, max_content_chars_per_run)
        self.max_cached_document_chars = max(1, max_cached_document_chars)
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_concurrency_per_domain = max(
            1, min(self.max_concurrency, int(max_concurrency_per_domain))
        )
        self.failure_ttl_seconds = max(0.1, float(failure_ttl_seconds))
        self.deterministic_failure_ttl_seconds = max(
            self.failure_ttl_seconds,
            float(deterministic_failure_ttl_seconds),
        )
        self.max_failures_per_run = max(1, int(max_failures_per_run))
        self._runs: dict[str, _FetchRunCache] = {}
        self._guard = asyncio.Lock()
        self._slots = asyncio.Semaphore(self.max_concurrency)
        self._domain_slots: dict[tuple[str, str], asyncio.Semaphore] = {}
        self._active_fetches = 0
        self._pending_fetches = 0
        self._peak_fetches = 0

    @staticmethod
    def _prune_failures(run_cache: _FetchRunCache, now: float) -> None:
        expired = [
            key
            for key, entry in run_cache.failures.items()
            if entry.expires_at <= now
        ]
        for key in expired:
            run_cache.failures.pop(key, None)

    @staticmethod
    def _raise_cached_failure(
        run_cache: _FetchRunCache, url_key: str, now: float
    ) -> None:
        entry = run_cache.failures.get(url_key)
        if entry is None or entry.expires_at <= now:
            return
        run_cache.failures.move_to_end(url_key)
        raise CachedFetchFailure(
            url_key,
            entry.error_type,
            entry.message,
        )

    async def _run_queued(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        waiting = True
        async with self._guard:
            self._pending_fetches += 1
        try:
            async with self._slots:
                async with self._guard:
                    self._pending_fetches -= 1
                    waiting = False
                    self._active_fetches += 1
                    self._peak_fetches = max(
                        self._peak_fetches, self._active_fetches
                    )
                try:
                    return await operation()
                finally:
                    async with self._guard:
                        self._active_fetches -= 1
        except BaseException:
            if waiting:
                async with self._guard:
                    self._pending_fetches -= 1
            raise

    async def _run_domain_queued(
        self,
        run_id: str,
        domain: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        if not domain:
            return await self._run_queued(operation)
        key = (run_id, domain)
        async with self._guard:
            domain_slot = self._domain_slots.setdefault(
                key,
                asyncio.Semaphore(self.max_concurrency_per_domain),
            )
        # Wait for the host-specific slot before taking a global slot, so a
        # busy domain cannot reserve capacity that other domains could use.
        async with domain_slot:
            return await self._run_queued(operation)

    async def claim_url(self, run_id: str, url_key: str, task_id: str) -> bool:
        """Give one worker exclusive extraction ownership of a canonical URL."""

        async with self._guard:
            run_cache = self._runs.setdefault(run_id, _FetchRunCache())
            if url_key in run_cache.url_owners:
                return False
            run_cache.url_owners[url_key] = task_id
            return True

    async def release_url(self, run_id: str, url_key: str, task_id: str) -> bool:
        """Release ownership after a fetch failure so a later round may retry."""

        async with self._guard:
            run_cache = self._runs.get(run_id)
            if (
                run_cache is None
                or run_cache.url_owners.get(url_key) != task_id
            ):
                return False
            run_cache.url_owners.pop(url_key, None)
            return True

    @staticmethod
    def _content_chars(value: Any) -> int:
        content = getattr(value, "content", None)
        if content is None and isinstance(value, dict):
            content = value.get("content")
        return len(content) if isinstance(content, str) else 0

    async def fetch(
        self,
        run_id: str,
        url_key: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        domain: str = "",
    ) -> tuple[Any, str]:
        """Return ``(document, mode)`` where mode is fetched/shared/cached."""

        async with self._guard:
            run_cache = self._runs.setdefault(run_id, _FetchRunCache())
            now = time.monotonic()
            self._prune_failures(run_cache, now)
            self._raise_cached_failure(run_cache, url_key, now)
            if url_key in run_cache.completed:
                value = run_cache.completed.pop(url_key)
                run_cache.completed[url_key] = value
                return value, "cached"
            task = run_cache.in_flight.get(url_key)
            owner = task is None
            if task is None:
                task = asyncio.create_task(
                    self._run_domain_queued(run_id, domain, operation)
                )
                run_cache.in_flight[url_key] = task

        try:
            value = await (task if owner else asyncio.shield(task))
        except Exception as exc:
            if owner:
                response = getattr(exc, "response", None)
                status = getattr(exc, "status_code", None)
                if status is None and response is not None:
                    status = getattr(response, "status_code", None)
                deterministic = (
                    status in {400, 401, 403, 404, 405, 410, 451}
                    or isinstance(exc, (PermissionError, ValueError))
                    or type(exc).__name__ in {"SecurityError"}
                )
                failure_ttl = (
                    self.deterministic_failure_ttl_seconds
                    if deterministic
                    else self.failure_ttl_seconds
                )
                async with self._guard:
                    current = self._runs.setdefault(run_id, _FetchRunCache())
                    current.failures[url_key] = _NegativeFetchEntry(
                        error_type=type(exc).__name__,
                        message=str(exc)[:500],
                        expires_at=time.monotonic() + failure_ttl,
                    )
                    current.failures.move_to_end(url_key)
                    while len(current.failures) > self.max_failures_per_run:
                        current.failures.popitem(last=False)
            raise
        finally:
            if owner:
                async with self._guard:
                    current = self._runs.get(run_id)
                    if current is not None and current.in_flight.get(url_key) is task:
                        current.in_flight.pop(url_key, None)

        if owner:
            size = self._content_chars(value)
            if size <= self.max_cached_document_chars:
                async with self._guard:
                    current = self._runs.setdefault(run_id, _FetchRunCache())
                    if url_key not in current.completed:
                        current.completed[url_key] = value
                        current.completed_chars += size
                    current.failures.pop(url_key, None)
                    while current.completed and (
                        len(current.completed) > self.max_entries_per_run
                        or current.completed_chars > self.max_content_chars_per_run
                    ):
                        _, evicted = current.completed.popitem(last=False)
                        current.completed_chars -= self._content_chars(evicted)
        return value, "fetched" if owner else "shared"

    async def lookup(self, run_id: str, url_key: str) -> tuple[Any, str] | None:
        """Reuse a completed or already in-flight URL without taking a lease."""

        async with self._guard:
            run_cache = self._runs.get(run_id)
            if run_cache is None:
                return None
            now = time.monotonic()
            self._prune_failures(run_cache, now)
            self._raise_cached_failure(run_cache, url_key, now)
            if url_key in run_cache.completed:
                value = run_cache.completed.pop(url_key)
                run_cache.completed[url_key] = value
                return value, "cached"
            task = run_cache.in_flight.get(url_key)
        if task is None:
            return None
        return await asyncio.shield(task), "shared"

    async def clear_run(self, run_id: str) -> None:
        async with self._guard:
            self._runs.pop(run_id, None)
            for key in [key for key in self._domain_slots if key[0] == run_id]:
                self._domain_slots.pop(key, None)

    async def snapshot(self, run_id: str) -> dict[str, int]:
        async with self._guard:
            current = self._runs.get(run_id)
            if current is None:
                return {
                    "completed": 0,
                    "in_flight": 0,
                    "content_chars": 0,
                    "active": self._active_fetches,
                    "pending": self._pending_fetches,
                    "peak": self._peak_fetches,
                    "max_concurrency": self.max_concurrency,
                }
            self._prune_failures(current, time.monotonic())
            return {
                "completed": len(current.completed),
                "in_flight": len(current.in_flight),
                "content_chars": current.completed_chars,
                "active": self._active_fetches,
                "pending": self._pending_fetches,
                "peak": self._peak_fetches,
                "max_concurrency": self.max_concurrency,
            }

    async def owner_snapshot(self, run_id: str) -> dict[str, str]:
        async with self._guard:
            current = self._runs.get(run_id)
            return dict(current.url_owners) if current is not None else {}


def allocate_task_budgets(run_budget: RunBudget, task_count: int) -> list[TaskBudget]:
    """Allocate the initial round research pool across independent workers."""
    task_count = max(1, task_count)
    search_base, search_extra = divmod(run_budget.max_total_searches, task_count)
    fetch_base, fetch_extra = divmod(run_budget.max_total_fetches, task_count)
    return [
        TaskBudget(
            max_searches=search_base + (1 if index < search_extra else 0),
            max_fetches=fetch_base + (1 if index < fetch_extra else 0),
            max_wall_time_seconds=run_budget.max_wall_time_seconds,
        )
        for index in range(task_count)
    ]


def remaining_task_budget(
    run_budget: RunBudget,
    events: list[dict],
    *,
    round_no: int = 1,
    max_rounds: int = 3,
) -> TaskBudget:
    """Return the global capacity not yet consumed by prior rounds.

    Per-worker passes remain bounded, so exposing the full remainder does not
    force a round to consume it. This avoids stranding a fixed future-round
    reserve when time or convergence prevents that round from being created.
    """
    usage = summarize_usage(events)
    remaining_searches = max(0, run_budget.max_total_searches - int(usage["searches"]))
    remaining_fetches = max(0, run_budget.max_total_fetches - int(usage["fetches"]))
    return TaskBudget(
        max_searches=remaining_searches,
        max_fetches=remaining_fetches,
        max_wall_time_seconds=run_budget.max_wall_time_seconds,
    )


def split_task_budget(pool: TaskBudget, task_count: int) -> list[TaskBudget]:
    """Split one round's remaining pool across the planner's worker tasks."""
    task_count = max(1, task_count)
    search_base, search_extra = divmod(pool.max_searches, task_count)
    fetch_base, fetch_extra = divmod(pool.max_fetches, task_count)
    return [
        TaskBudget(
            max_searches=search_base + (1 if index < search_extra else 0),
            max_fetches=fetch_base + (1 if index < fetch_extra else 0),
            max_wall_time_seconds=pool.max_wall_time_seconds,
        )
        for index in range(task_count)
    ]


def task_usage(events: list[dict], task_id: str) -> dict[str, int]:
    selected = [event for event in events if event.get("task_id") == task_id]
    return {
        "tokens": sum(event.get("input_tokens", 0) + event.get("output_tokens", 0) for event in selected),
        "searches": sum(event.get("searches", 0) for event in selected),
        "fetches": sum(event.get("fetches", 0) for event in selected),
    }


def summarize_usage(events: list[dict]) -> dict[str, int | float | None]:
    unique = {event["usage_event_id"]: event for event in events}
    return {
        "input_tokens": sum(event.get("input_tokens", 0) for event in unique.values()),
        "output_tokens": sum(event.get("output_tokens", 0) for event in unique.values()),
        "total_tokens": sum(
            event.get("input_tokens", 0) + event.get("output_tokens", 0)
            for event in unique.values()
        ),
        "searches": sum(event.get("searches", 0) for event in unique.values()),
        "fetches": sum(event.get("fetches", 0) for event in unique.values()),
        "estimated_cost_usd": None,
    }


def usage_event(
    run_id: str,
    task_id: str | None,
    category: str,
    sequence: int,
    **values: int,
) -> UsageEvent:
    return UsageEvent(
        usage_event_id=make_id("use", run_id, task_id, category, sequence),
        run_id=run_id,
        task_id=task_id,
        category=category,
        timestamp=utc_now(),
        **values,
    )
