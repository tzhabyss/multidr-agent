"""Command-line entry point for one-off Deep Research runs."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from .schemas import (
    ResearchConstraints,
    ResearchMode,
    ResearchOptions,
    ResearchRequest,
    RunStatus,
    TaskSufficiencyOverrides,
)
from .service import ResearchService, RunAlreadyExistsError, RunNotFoundError
from .interactive import run_terminal_shell
from .skill_loader import discover_skills
from .runtime_config import load_runtime_config, resolve_runtime_config_path


def _configure_stdio_utf8() -> None:
    """Prevent valid research text from failing on legacy Windows code pages."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        encoding = str(getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding in {"utf8", "cp65001"}:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            # Embedded callers may expose immutable or already-closed streams.
            # Their own output policy remains in effect.
            continue


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepresearch",
        description="Evidence-driven multi-round Deep Research agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    knowledge = subparsers.add_parser("knowledge", help="import or search the local research library")
    knowledge.add_argument("action", choices=("ingest", "search", "reindex", "jobs", "retry", "reconcile", "migrate"))
    knowledge.add_argument("value", nargs="?", help="file path or search query")
    knowledge.add_argument("--config", type=Path)
    knowledge.add_argument("--db", type=Path)
    knowledge.add_argument("--source-config", type=Path, help="legacy mongo_milvus config for explicit migration")
    knowledge.add_argument("--collection")
    knowledge.add_argument("--document-id", help="stable logical identity across renamed versions")
    knowledge.add_argument("--reparse", action="store_true", help="reparse unchanged bytes and publish a new version (Milvus backends)")
    knowledge.add_argument("--as-of", help="snapshot capture cutoff, ISO timestamp with timezone")
    knowledge.add_argument("--lexical-only", action="store_true")
    monitor = subparsers.add_parser("monitor", help="summarize persisted run operations without external calls")
    monitor.add_argument("run_dir", type=Path)
    trace = subparsers.add_parser("trace", help="read persisted run operations")
    trace.add_argument("run_dir", type=Path)

    doctor = subparsers.add_parser("doctor", help="check local runtime configuration")
    doctor.add_argument("--config", type=Path)

    web = subparsers.add_parser("web", help="start the local Deep Research web console")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--artifact-root", type=Path)
    web.add_argument("--checkpoint-db", type=Path)
    web.add_argument("--config", type=Path)

    run = subparsers.add_parser("run", help="run a research request")
    run.add_argument("question", help="research question")
    run.add_argument("--profile", default="general_research")
    run.add_argument("--run-id")
    run.add_argument("--source-url", action="append", default=[])
    run.add_argument("--required-dimension", action="append", default=[])
    run.add_argument("--exclude-dimension", action="append", default=[])
    run.add_argument("--allow-domain", action="append", default=[])
    run.add_argument("--block-domain", action="append", default=[])
    run.add_argument("--date-boundary")
    run.add_argument("--locale", default="zh-CN")
    run.add_argument("--mode", choices=("quick", "standard", "deep", "max"), default="standard")
    run.add_argument("--max-rounds", type=int, choices=range(1, 7))
    run.add_argument("--max-worker-iterations", type=int, choices=range(1, 13))
    run.add_argument("--max-workers", type=int, choices=range(1, 17))
    run.add_argument("--min-queries", type=int)
    run.add_argument("--min-query-coverage", type=int)
    run.add_argument("--min-successful-extractions", type=int)
    run.add_argument("--min-effective-sources", type=int)
    run.add_argument("--min-independent-domains", type=int)
    run.add_argument("--min-evidence", type=int)
    run.add_argument("--max-source-quality-tier", type=int, choices=range(1, 5))
    run.add_argument("--artifact-root", type=Path)
    run.add_argument("--checkpoint-db", type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--json", action="store_true", dest="as_json")

    chat = subparsers.add_parser("chat", help="send one turn to a persistent session")
    chat.add_argument("message", help="user message")
    chat.add_argument("--session-id")
    chat.add_argument("--user-id", default="local-user")
    chat.add_argument("--profile", default="general_research")
    chat.add_argument("--locale", default="zh-CN")
    chat.add_argument("--mode", choices=("quick", "standard", "deep", "max"), default="standard")
    chat.add_argument("--max-rounds", type=int, choices=range(1, 7))
    chat.add_argument("--max-worker-iterations", type=int, choices=range(1, 13))
    chat.add_argument("--max-workers", type=int, choices=range(1, 17))
    chat.add_argument("--artifact-root", type=Path)
    chat.add_argument("--checkpoint-db", type=Path)
    chat.add_argument("--config", type=Path)
    chat.add_argument("--json", action="store_true", dest="as_json")

    shell = subparsers.add_parser("shell", help="start an interactive research session")
    shell.add_argument("--session-id")
    shell.add_argument("--user-id", default="local-user")
    shell.add_argument("--profile", default="general_research")
    shell.add_argument("--locale", default="zh-CN")
    shell.add_argument("--mode", choices=("quick", "standard", "deep", "max"), default="standard")
    shell.add_argument("--max-rounds", type=int, choices=range(1, 7))
    shell.add_argument("--max-worker-iterations", type=int, choices=range(1, 13))
    shell.add_argument("--max-workers", type=int, choices=range(1, 17))
    shell.add_argument("--artifact-root", type=Path)
    shell.add_argument("--checkpoint-db", type=Path)
    shell.add_argument("--config", type=Path)
    shell.add_argument(
        "--progress", choices=("quiet", "normal", "verbose"), default="normal"
    )
    shell.add_argument("--heartbeat-seconds", type=float, default=15.0)

    resume = subparsers.add_parser("resume", help="resume or read a checkpointed run")
    resume.add_argument("run_id")
    resume.add_argument("--artifact-root", type=Path)
    resume.add_argument("--checkpoint-db", type=Path)
    resume.add_argument("--config", type=Path)
    resume.add_argument("--json", action="store_true", dest="as_json")

    finalize = subparsers.add_parser(
        "finalize-only",
        help="rerun Writer, citations, and artifacts from an existing checkpoint",
    )
    finalize.add_argument("run_id")
    finalize.add_argument("--artifact-root", type=Path)
    finalize.add_argument("--checkpoint-db", type=Path)
    finalize.add_argument("--config", type=Path)
    finalize.add_argument("--json", action="store_true", dest="as_json")
    return parser


def build_request(args: argparse.Namespace) -> ResearchRequest:
    constraints = ResearchConstraints(
        locale=args.locale,
        date_boundary=args.date_boundary,
        allowed_domains=args.allow_domain,
        blocked_domains=args.block_domain,
        required_dimensions=args.required_dimension,
        excluded_dimensions=args.exclude_dimension,
    )
    sufficiency = TaskSufficiencyOverrides(
        min_queries=args.min_queries,
        min_query_coverage=args.min_query_coverage,
        min_successful_extractions=args.min_successful_extractions,
        min_effective_sources=args.min_effective_sources,
        min_independent_domains=args.min_independent_domains,
        min_evidence=args.min_evidence,
        max_source_quality_tier=args.max_source_quality_tier,
    )
    option_values = {
        "task_sufficiency": sufficiency,
    }
    for key in ("max_rounds", "max_worker_iterations", "max_workers"):
        value = getattr(args, key)
        if value is not None:
            option_values[key] = value
    return ResearchRequest(
        question=args.question,
        mode=ResearchMode(args.mode),
        profile_id=args.profile,
        source_urls=args.source_url,
        constraints=constraints,
        options=ResearchOptions(**option_values),
        run_id=args.run_id,
    )


def _doctor(config_path: Path | None = None) -> int:
    packages = [
        "langgraph",
        "langgraph-checkpoint-sqlite",
        "openai",
        "pydantic",
        "httpx",
        "mcp",
    ]
    rows: list[tuple[str, bool, str]] = []
    runtime_config = None
    try:
        runtime_config = load_runtime_config(config_path)
        tokens = runtime_config.llm_output_tokens
        rows.append(
            (
                "runtime config",
                True,
                f"{resolve_runtime_config_path(config_path)}; "
                f"timezone={runtime_config.timezone}; "
                f"mode={runtime_config.default_mode}; "
                f"synthesizer={tokens.synthesizer_assessment}, "
                f"report={tokens.report_writer}",
            )
        )
        routing = runtime_config.mode_config(runtime_config.default_mode).stages
        rows.append(
            (
                "model routing",
                True,
                (
                    f"planner={routing.planner}; "
                    f"worker={routing.worker_agent_loop}; "
                    f"synthesizer={routing.synthesizer}; "
                    f"writer={routing.report_writer}; "
                    f"tool_roles={','.join(runtime_config.tool_permissions.executable_roles)}"
                ),
            )
        )
    except Exception as exc:
        rows.append(("runtime config", False, f"{type(exc).__name__}: {exc}"))
    if runtime_config is not None:
        enabled_servers = [
            server
            for server in runtime_config.mcp.servers.values()
            if runtime_config.mcp.enabled and server.runtime_enabled()
        ]
        allowlisted_tools = sum(
            1
            for server in enabled_servers
            for tool in server.tools.values()
            if tool.enabled
        )
        rows.append(
            (
                "mcp config",
                True,
                (
                    f"enabled={runtime_config.mcp.enabled}; "
                    f"active_servers={len(enabled_servers)}/"
                    f"{len(runtime_config.mcp.servers)}; "
                    f"allowlisted_tools={allowlisted_tools}"
                ),
            )
        )
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            rows.append((package, True, version))
        except importlib.metadata.PackageNotFoundError:
            rows.append((package, False, "not installed"))

    provider_key_rows: list[tuple[str, bool, str]] = []
    if runtime_config is not None:
        active_provider_ids = {
            tier.provider for tier in runtime_config.model_tiers.values()
        }
        key_envs = sorted(
            {
                runtime_config.llm_providers[provider_id].api_key_env
                for provider_id in active_provider_ids
            }
        )
        for key_env in key_envs:
            present = bool(os.getenv(key_env))
            provider_key_rows.append(
                (key_env, present, "configured" if present else "missing")
            )
    else:
        present = bool(os.getenv("OPENAI_API_KEY"))
        provider_key_rows.append(
            ("OPENAI_API_KEY", present, "configured" if present else "missing")
        )
    try:
        skill_definitions = discover_skills(Path(__file__).resolve().parent.parent)
        skill_by_id = {item.skill_id: item for item in skill_definitions}
        web_search = skill_by_id.get("web-search")
        provider_chain = [
            name
            for name in (web_search.fallback_skills if web_search is not None else ())
            if name in skill_by_id and skill_by_id[name].enabled
        ]
        disabled = [item.skill_id for item in skill_definitions if not item.enabled]
        skill_detail = (
            f"{len(skill_definitions)} discovered; web-search -> "
            + " -> ".join(provider_chain)
        )
        if disabled:
            skill_detail += "; disabled: " + ", ".join(disabled)
        skill_row = ("skills", True, skill_detail)
    except Exception as exc:
        skill_row = ("skills", False, f"{type(exc).__name__}: {exc}")
    rows.extend(
        [
            *provider_key_rows,
            ("OPENAI_BASE_URL", True, os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            ("OPENAI_MODEL", True, os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
            skill_row,
            ("artifact root", True, os.getenv("DEEPRESEARCH_ARTIFACT_ROOT", "artifacts")),
            (
                "checkpoint DB",
                True,
                os.getenv("DEEPRESEARCH_CHECKPOINT_DB", ".runtime/checkpoints.sqlite"),
            ),
        ]
    )
    for name, ok, detail in rows:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in rows) else 1


def _emit_result(result, *, as_json: bool) -> int:
    summary = {
        "run_id": result.run_id,
        "status": result.status.value,
        "mode": result.mode.value,
        "citation_gate_passed": result.gate.passed,
        "gate_violations": result.gate.violations,
        "usage": result.usage,
        "source_count": len(result.sources),
        "claim_count": len(result.claims),
        "artifact_dir": result.artifact_dir,
        "output_path": result.output_path,
        "audit_report_path": result.audit_report_path,
        "audit_log_path": result.audit_log_path,
    }
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"run_id: {result.run_id}")
        print(f"status: {result.status.value}")
        print(f"citation_gate: {'passed' if result.gate.passed else 'failed'}")
        print(f"sources: {len(result.sources)}; claims: {len(result.claims)}")
        print(f"usage: {json.dumps(result.usage, ensure_ascii=False)}")
        print(f"output: {result.output_path}")
        print(f"audit_report: {result.audit_report_path}")
        print(f"audit_log: {result.audit_log_path}")
        if result.gate.violations:
            print("gate_violations:")
            for violation in result.gate.violations:
                print(f"- {violation}")

    if result.status == RunStatus.COMPLETED_VERIFIED:
        return 0
    if result.status == RunStatus.COMPLETED_PARTIAL:
        return 3
    return 1


async def _run(args: argparse.Namespace) -> int:
    request = build_request(args)
    async with ResearchService(
        artifact_root=args.artifact_root,
        checkpoint_path=args.checkpoint_db,
        config_path=args.config,
    ) as service:
        result = await service.run(request)
    return _emit_result(result, as_json=args.as_json)


async def _resume(args: argparse.Namespace) -> int:
    async with ResearchService(
        artifact_root=args.artifact_root,
        checkpoint_path=args.checkpoint_db,
        config_path=args.config,
    ) as service:
        result = await service.resume(args.run_id)
    return _emit_result(result, as_json=args.as_json)


async def _finalize_only(args: argparse.Namespace) -> int:
    async with ResearchService(
        artifact_root=args.artifact_root,
        checkpoint_path=args.checkpoint_db,
        config_path=args.config,
    ) as service:
        result = await service.finalize_only(args.run_id)
    return _emit_result(result, as_json=args.as_json)


async def _chat(args: argparse.Namespace) -> int:
    option_overrides = {
        key: getattr(args, key)
        for key in ("max_rounds", "max_worker_iterations", "max_workers")
        if getattr(args, key) is not None
    }
    overrides = {
        "mode": args.mode,
        "profile_id": args.profile,
        "constraints": {"locale": args.locale},
        "options": option_overrides,
    }
    async with ResearchService(
        artifact_root=args.artifact_root,
        checkpoint_path=args.checkpoint_db,
        config_path=args.config,
    ) as service:
        result = await service.chat(
            args.message,
            session_id=args.session_id,
            user_id=args.user_id,
            request_overrides=overrides,
        )
    payload = result.model_dump(mode="json")
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"session_id: {result.session_id}")
        print(f"turn_id: {result.turn_id}")
        print(f"action: {result.action}")
        if result.run is not None:
            print(f"run_id: {result.run.run_id}")
        print(result.message)
    if result.run is None:
        return 0
    return 0 if result.run.status == RunStatus.COMPLETED_VERIFIED else 3


async def _shell(args: argparse.Namespace) -> int:
    option_overrides = {
        key: getattr(args, key)
        for key in ("max_rounds", "max_worker_iterations", "max_workers")
        if getattr(args, key) is not None
    }
    overrides = {
        "mode": args.mode,
        "profile_id": args.profile,
        "constraints": {"locale": args.locale},
        "options": option_overrides,
    }
    async with ResearchService(
        artifact_root=args.artifact_root,
        checkpoint_path=args.checkpoint_db,
        config_path=args.config,
    ) as service:
        service.runtime_config.profile_config(args.profile)
        return await run_terminal_shell(
            service,
            user_id=args.user_id,
            session_id=args.session_id,
            request_overrides=overrides,
            progress_level=args.progress,
            heartbeat_seconds=max(0.0, args.heartbeat_seconds),
        )


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio_utf8()
    args = _parser().parse_args(argv)
    try:
        if args.command == "knowledge":
            from .knowledge import create_store
            from .memory_system.embeddings import HashEmbeddingProvider
            config = load_runtime_config(args.config).knowledge
            store = create_store(config, path=args.db,
                                 provider=HashEmbeddingProvider() if args.lexical_only else None)
            try:
                if args.action == "search" and config.reranker_model and not args.lexical_only:
                    from .knowledge.reranker import CrossEncoderReranker
                    store.reranker = CrossEncoderReranker.from_config(config)
                collection = args.collection or config.collection
                if args.action in {"ingest", "search", "retry"} and not args.value:
                    raise ValueError("knowledge ingest/search requires a file path or query")
                if args.action in {"jobs", "retry", "reconcile"} and config.backend not in {"milvus", "mongo_milvus"}:
                    raise ValueError("jobs/retry/reconcile require milvus or legacy mongo_milvus")
                if args.document_id and config.backend not in {"milvus", "mongo_milvus"}:
                    raise ValueError("--document-id requires milvus or legacy mongo_milvus")
                if args.reparse and (args.action != "ingest" or config.backend not in {"milvus", "mongo_milvus"}):
                    raise ValueError("--reparse requires knowledge ingest with a Milvus backend")
                if args.action == "migrate":
                    if config.backend != "milvus" or args.source_config is None:
                        raise ValueError("migrate requires target backend milvus and --source-config")
                    source_config = load_runtime_config(args.source_config).knowledge
                    if source_config.backend != "mongo_milvus":
                        raise ValueError("--source-config must select the legacy mongo_milvus backend")
                    from .knowledge.migration import migrate_collection
                    source = create_store(source_config, provider=store.provider)
                    try:
                        result = migrate_collection(source, store, collection=collection)
                    finally:
                        source.close()
                elif args.action == "ingest":
                    kwargs = {"document_id": args.document_id} if args.document_id else {}
                    if args.reparse:
                        kwargs["reparse"] = True
                    result = store.ingest_file(args.value, collection=collection, **kwargs)
                elif args.action == "search":
                    result = store.search(args.value, collection=collection, as_of=args.as_of,
                                          mode="bm25" if args.lexical_only else "hybrid")
                elif args.action == "retry":
                    result = store.retry(args.value)
                else:
                    result = getattr(store, args.action)(collection=collection)
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                return 0
            finally:
                if callable(getattr(store, "close", None)):
                    store.close()
        if args.command == "web":
            import uvicorn

            from .web.app import create_app

            app = create_app(
                artifact_root=args.artifact_root,
                checkpoint_path=args.checkpoint_db,
                config_path=args.config,
            )
            uvicorn.run(app, host=args.host, port=args.port)
            return 0
        if args.command in {"monitor", "trace"}:
            from .runtime_monitor import RuntimeMonitor
            from .tool_runtime import ToolRuntime
            if not args.run_dir.is_dir():
                raise ValueError(f"Run directory does not exist: {args.run_dir}")
            rows = []
            for path in sorted((args.run_dir / "details" / "runtime").glob("operations-*.jsonl")):
                rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            rows.sort(key=lambda row: row["finished_at"])
            value = rows if args.command == "trace" else RuntimeMonitor(ToolRuntime()).summary(rows)
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        if args.command == "doctor":
            return _doctor(args.config)
        if args.command == "resume":
            return asyncio.run(_resume(args))
        if args.command == "finalize-only":
            return asyncio.run(_finalize_only(args))
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "shell":
            return asyncio.run(_shell(args))
        return asyncio.run(_run(args))
    except (ValidationError, ValueError, RunAlreadyExistsError, RunNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except asyncio.CancelledError:
        print("cancelled by user", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print("cancelled by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
