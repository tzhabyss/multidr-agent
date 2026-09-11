"""Call task44's craft Planner directly and preserve the unparsed response.

This diagnostic intentionally bypasses ModelRouter recovery, JSON parsing,
Pydantic validation, DAG validation, and trace slicing.  It is used to isolate
provider/model behaviour from application-level Planner acceptance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PATH = ROOT / "docs" / "task44-craft-planner-system-20260812.txt"
USER_PATH = ROOT / "docs" / "task44-craft-planner-user-20260812.json"


def now_local() -> str:
    return datetime.now().astimezone().isoformat()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text", "json"), required=True)
    parser.add_argument("--max-tokens", type=int, default=40_000)
    parser.add_argument("--timeout-seconds", type=float, default=1_800)
    parser.add_argument(
        "--prompt-variant", choices=("full", "short", "combined"), default="full"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider-file", type=Path)
    args = parser.parse_args()

    provider_values: dict[str, str] = {}
    if args.provider_file:
        for raw_line in args.provider_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = raw_line.partition(":")
            if separator:
                provider_values[key.strip().casefold()] = value.strip()
    load_dotenv(ROOT / ".env")
    api_key = provider_values.get("key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    base_url = provider_values.get("url") or os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    if base_url.rstrip("/").casefold().endswith("api.deepseek.com"):
        base_url = base_url.rstrip("/") + "/v1"
    model = (
        provider_values.get("model name")
        or os.getenv("DEEPRESEARCH_CRAFT_MODEL")
        or os.getenv("OPENAI_MODEL")
    )
    if not model:
        raise RuntimeError("craft model is not configured")

    if args.prompt_variant == "short":
        system = "Return a concise answer."
        user = "Reply with exactly: OK"
    else:
        system = SYSTEM_PATH.read_text(encoding="utf-8")
        user = USER_PATH.read_text(encoding="utf-8")
    request_system = system
    messages: list[dict[str, str]] = [
        {"role": "system", "content": request_system},
        {"role": "user", "content": user},
    ]
    if args.prompt_variant == "combined":
        # Keep the complete bytes in the request while isolating the gateway's
        # handling of a very large system-role message.
        request_system = "You are a helpful assistant."
        user = f"{system}\n\n--- TASK PAYLOAD ---\n\n{user}"
        messages = [{"role": "user", "content": user}]
    request_options: dict[str, object] = {}
    if args.mode == "json":
        request_system += "\nReturn exactly one valid JSON object and no Markdown."
        request_options["response_format"] = {"type": "json_object"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.prompt_variant}-{args.mode}-{args.max_tokens}"
    raw_path = args.output_dir / f"raw-{suffix}.txt"
    metadata_path = args.output_dir / f"metadata-{suffix}.json"
    request_path = args.output_dir / f"request-{suffix}.json"
    request_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "prompt_variant": args.prompt_variant,
                "model": model,
                "temperature": 0.1,
                "max_tokens": args.max_tokens,
                "timeout_seconds": args.timeout_seconds,
                "system_chars": len(request_system),
                "user_chars": len(user),
                "message_roles": [message["role"] for message in messages],
                "response_format": request_options.get("response_format"),
                "started_at": now_local(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    started_wall = now_local()
    started = time.monotonic()
    metadata: dict[str, object] = {
        "mode": args.mode,
        "prompt_variant": args.prompt_variant,
        "model_requested": model,
        "started_at": started_wall,
        "timeout_seconds": args.timeout_seconds,
        "max_tokens": args.max_tokens,
        "response_format": request_options.get("response_format"),
    }
    print(
        json.dumps(
            {
                "event": "request_started",
                "mode": args.mode,
                "model": model,
                "started_at": started_wall,
                "max_tokens": args.max_tokens,
                "timeout_seconds": args.timeout_seconds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=args.max_tokens,
            **request_options,
        )
        duration = time.monotonic() - started
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        usage = response.usage
        raw_path.write_text(content, encoding="utf-8")
        metadata.update(
            {
                "status": "completed",
                "ended_at": now_local(),
                "duration_seconds": round(duration, 3),
                "model_returned": getattr(response, "model", None),
                "finish_reason": choice.finish_reason,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        duration = time.monotonic() - started
        metadata.update(
            {
                "status": "failed",
                "ended_at": now_local(),
                "duration_seconds": round(duration, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        return 1
    finally:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
