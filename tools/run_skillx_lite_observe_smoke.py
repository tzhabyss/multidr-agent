#!/usr/bin/env python3
"""Run one real SkillX-Lite observe smoke test under a shared token cap."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepresearch.llm import LLMJSONError, LLMResponse, ResearchLLM
from deepresearch.model_routing import ModelRouter
from deepresearch.retrieval import (
    DuckDuckGoSearchBackend,
    FallbackSearchBackend,
    SafeWebFetcher,
    WikipediaSearchBackend,
)
from deepresearch.runtime_config import load_runtime_config
from deepresearch.schemas import (
    ResearchConstraints,
    ResearchMode,
    ResearchOptions,
    ResearchRequest,
    RunBudget,
)
from deepresearch.service import ResearchService


T = TypeVar("T")


class TokenCapExceeded(RuntimeError):
    """Raised before an LLM call that could exceed the shared hard cap."""


class TokenCappedLLM:
    """Conservatively reserve shared tokens, then settle from provider usage.

    UTF-8 byte length is used as an intentionally conservative input estimate.
    An additional per-call allowance covers chat-message framing. Because the
    provider's requested completion limit is reserved in full, concurrent calls
    cannot collectively cross ``hard_limit`` under this guard.
    """

    def __init__(
        self,
        delegate: ResearchLLM,
        *,
        hard_limit: int,
        framing_allowance: int = 4096,
    ) -> None:
        self.delegate = delegate
        self.model_id = delegate.model_id
        self.provider_id = getattr(delegate, "provider_id", "injected")
        self.hard_limit = hard_limit
        self.framing_allowance = framing_allowance
        self._lock = asyncio.Lock()
        self._reserved = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls: list[dict[str, Any]] = []

    @staticmethod
    def _payload_bytes(payload: Any) -> int:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return len(encoded)

    async def _reserve(self, kind: str, payload: Any, max_tokens: int) -> int:
        reservation = (
            self._payload_bytes(payload)
            + int(max_tokens)
            + self.framing_allowance
        )
        async with self._lock:
            committed = self._input_tokens + self._output_tokens
            projected = committed + self._reserved + reservation
            if projected > self.hard_limit:
                raise TokenCapExceeded(
                    f"LLM token cap would be exceeded: committed={committed}, "
                    f"reserved={self._reserved}, next={reservation}, "
                    f"hard_limit={self.hard_limit}, kind={kind}"
                )
            self._reserved += reservation
            call_no = len(self._calls) + 1
            print(
                json.dumps(
                    {
                        "event": "llm_call_reserved",
                        "call": call_no,
                        "kind": kind,
                        "max_output_tokens": max_tokens,
                        "reservation": reservation,
                        "projected_total": projected,
                        "hard_limit": self.hard_limit,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return reservation

    async def _settle(
        self,
        *,
        kind: str,
        reservation: int,
        response: LLMResponse | None,
        error: BaseException | None = None,
    ) -> None:
        input_tokens = int(response.input_tokens if response is not None else 0)
        output_tokens = int(response.output_tokens if response is not None else 0)
        async with self._lock:
            self._reserved -= reservation
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            row = {
                "call": len(self._calls) + 1,
                "kind": kind,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "error": type(error).__name__ if error is not None else None,
            }
            self._calls.append(row)
            print(
                json.dumps(
                    {
                        "event": "llm_call_settled",
                        **row,
                        "committed_total": self._input_tokens
                        + self._output_tokens,
                        "outstanding_reservations": self._reserved,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    async def _invoke(
        self,
        kind: str,
        payload: Any,
        max_tokens: int,
        call: Callable[[], Awaitable[T]],
        response_of: Callable[[T], LLMResponse],
    ) -> T:
        reservation = await self._reserve(kind, payload, max_tokens)
        try:
            result = await call()
        except LLMJSONError as exc:
            await self._settle(
                kind=kind,
                reservation=reservation,
                response=exc.response,
                error=exc,
            )
            raise
        except BaseException as exc:
            await self._settle(
                kind=kind,
                reservation=reservation,
                response=None,
                error=exc,
            )
            raise
        response = response_of(result)
        await self._settle(
            kind=kind,
            reservation=reservation,
            response=response,
        )
        return result

    async def complete_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
    ) -> LLMResponse:
        return await self._invoke(
            "text",
            {"system": system, "user": user},
            max_tokens,
            lambda: self.delegate.complete_text(system, user, max_tokens=max_tokens),
            lambda result: result,
        )

    async def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
    ) -> tuple[dict[str, Any], LLMResponse]:
        return await self._invoke(
            "json",
            {"system": system, "user": user},
            max_tokens,
            lambda: self.delegate.complete_json(system, user, max_tokens=max_tokens),
            lambda result: result[1],
        )

    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        return await self._invoke(
            "tools",
            {"messages": messages, "tools": tools},
            max_tokens,
            lambda: self.delegate.complete_with_tools(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            ),
            lambda result: result,
        )

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            total = self._input_tokens + self._output_tokens
            return {
                "model_id": self.model_id,
                "provider_id": self.provider_id,
                "hard_limit": self.hard_limit,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": total,
                "remaining_tokens": self.hard_limit - total - self._reserved,
                "outstanding_reservations": self._reserved,
                "calls": list(self._calls),
            }


class FailOnceFetcher:
    """Inject one deterministic route failure, then delegate normally."""

    def __init__(self, delegate: SafeWebFetcher, url_substring: str) -> None:
        self.delegate = delegate
        self.url_substring = url_substring
        self.injected = False

    async def fetch(
        self,
        url: str,
        title: str | None = None,
        render: bool = False,
    ):
        if not self.injected and self.url_substring in url:
            self.injected = True
            raise RuntimeError(
                f"SkillX-Lite smoke-test injected fetch failure for {url}"
            )
        return await self.delegate.fetch(url, title=title, render=render)


def controlled_runtime_config(store_root: str):
    config = load_runtime_config()
    tokens = config.llm_output_tokens.model_copy(
        update={
            "planner": 6000,
            "worker_strategy": 1800,
            "worker_reflection": 1000,
            "worker_agent_loop": 1800,
            "evidence_extraction_min": 256,
            "evidence_extraction_max": 2200,
            "synthesizer_assessment": 2400,
            "report_writer": 4000,
            "reviewer": 2200,
            "extraction_context": 10000,
        }
    )
    graph = config.graph_engineering.model_copy(
        update={
            "max_plan_tasks": 3,
            "wave_size": 1,
            "extraction_max_concurrency": 1,
            "candidate_selection_max_candidates": 12,
        }
    )
    return config.model_copy(
        update={
            "llm_output_tokens": tokens,
            "graph_engineering": graph,
            "skill_evolution": config.skill_evolution.model_copy(
                update={"mode": "observe", "store_root": store_root}
            ),
            "memory": config.memory.model_copy(
                update={"enabled": False, "automatic_capture_enabled": False}
            ),
        }
    )


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"skillx-lite-smoke-{uuid.uuid4().hex[:10]}"
    config = controlled_runtime_config(args.store_root)
    bootstrap_router = ModelRouter(config)
    delegate = bootstrap_router.resolve_tier("lite")
    capped_llm = TokenCappedLLM(delegate, hard_limit=args.token_cap)
    request = ResearchRequest(
        run_id=run_id,
        question=(
            "核验论文 SkillX: Automatically Constructing Skill Knowledge Bases "
            "for Agents 的三级技能名称与职责、主要评测基准，以及官方代码仓库。"
            "优先使用论文原文、arXiv 和作者官方 GitHub；明确区分论文结论与推断，"
            "并列出无法从一手来源核实的事项。"
        ),
        mode=ResearchMode.QUICK,
        source_urls=[
            "https://arxiv.org/abs/2604.04804",
            "https://github.com/zjunlp/SkillX",
        ],
        constraints=ResearchConstraints(
            locale="zh-CN",
            allowed_domains=["arxiv.org", "github.com"],
            required_dimensions=["三级技能", "评测基准", "官方代码仓库"],
        ),
        options=ResearchOptions(
            max_rounds=1,
            max_worker_iterations=2,
            max_workers=1,
        ),
    )
    budget = RunBudget(
        max_total_tokens=args.token_cap,
        max_total_searches=4,
        max_total_fetches=8,
        max_wall_time_seconds=600,
    )
    started = time.monotonic()
    error: BaseException | None = None
    result = None
    base_fetcher = SafeWebFetcher(
        timeout_seconds=20,
        retries=1,
        allow_pdf=True,
        reader_fallback=True,
        academic_fallback=True,
    )
    fetcher = (
        FailOnceFetcher(base_fetcher, args.fail_once_url_substring)
        if args.fail_once_url_substring
        else base_fetcher
    )
    try:
        async with ResearchService.in_memory(
            llm=capped_llm,
            search_backend=FallbackSearchBackend(
                [DuckDuckGoSearchBackend(), WikipediaSearchBackend()]
            ),
            fetcher=fetcher,
            artifact_root=ROOT / "artifacts",
            runtime_config=config,
        ) as service:
            result = await service.run(request, budget=budget)
    except BaseException as exc:
        error = exc
    token_snapshot = await capped_llm.snapshot()
    artifact_dir = ROOT / "artifacts" / run_id
    summary = {
        "run_id": run_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "status": result.status.value if result is not None else "error",
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
        "token_guard": token_snapshot,
        "reported_usage": result.usage if result is not None else None,
        "artifact_dir": str(artifact_dir),
        "output_path": result.output_path if result is not None else None,
        "audit_report_path": result.audit_report_path if result is not None else None,
        "gate": result.gate.model_dump(mode="json") if result is not None else None,
        "fault_injection": {
            "fail_once_url_substring": args.fail_once_url_substring,
            "injected": bool(getattr(fetcher, "injected", False)),
        },
    }
    output_dir = ROOT / "output" / "skillx-lite-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_id}-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    print(json.dumps({"event": "smoke_complete", **summary}, ensure_ascii=False), flush=True)
    if error is not None:
        raise error
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--token-cap", type=int, default=300_000)
    parser.add_argument(
        "--store-root",
        default=".runtime/skill-evolution-smoke",
    )
    parser.add_argument("--fail-once-url-substring", default="")
    args = parser.parse_args()
    if args.token_cap <= 0 or args.token_cap > 1_000_000:
        parser.error("--token-cap must be between 1 and 1,000,000")
    asyncio.run(run_smoke(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
