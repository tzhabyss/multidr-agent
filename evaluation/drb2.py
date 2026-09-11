"""DeepResearch Bench II dataset adapter with rubric isolation."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from deepresearch.date_constraints import extract_date_boundary


DRB2_DATASET_URL = (
    "https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/"
    "main/tasks_and_rubrics.jsonl"
)


def normalize_drb2_record(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content") or {}
    rubric = content.get("rubric") or {}
    blocked = content.get("blocked") or {}
    language = str(record.get("language") or "en")
    atomic_rubrics: list[dict[str, Any]] = []
    for category in ("info_recall", "analysis", "presentation"):
        for index, criterion in enumerate(rubric.get(category, []), 1):
            atomic_rubrics.append(
                {
                    "rubric_id": f"{category}-{index:03d}",
                    "category": category,
                    "criterion": str(criterion),
                }
            )
    question = str(content.get("task") or record.get("prompt") or "").strip()
    date_boundary = extract_date_boundary(question)
    constraints: dict[str, Any] = {
        "locale": "zh-CN" if language == "zh" else "en",
        "blocked_urls": list(blocked.get("urls") or []),
    }
    if date_boundary:
        constraints["date_boundary"] = date_boundary
    return {
        "case_id": str(record.get("id") or record.get("idx")),
        "question": question,
        "mode": "standard",
        "profile_id": "drb_evaluation",
        "constraints": constraints,
        "atomic_rubrics": atomic_rubrics,
        "benchmark": "drb2",
        "benchmark_index": record.get("idx"),
        "language": language,
        "theme": record.get("theme"),
        "description": record.get("description"),
        "blocked_reference": {
            "title": blocked.get("title"),
            "authors": blocked.get("authors") or [],
            "urls": blocked.get("urls") or [],
        },
        "license": record.get("license"),
    }


async def _download(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            timeout = httpx.Timeout(90.0, connect=30.0)
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                trust_env=True,
                headers={"User-Agent": "EvidenceDeepResearch-Eval/0.1"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to download DRB II dataset: {last_error!r}")


async def prepare_drb2_dataset(
    destination: str | Path,
    *,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    source_url: str = DRB2_DATASET_URL,
) -> Path:
    """Download, select, and normalize official DRB II cases into local JSONL."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    raw = await _download(source_url)
    requested = set(case_ids or [])
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid official DRB II JSONL at line {line_number}") from exc
        case_id = str(record.get("id") or record.get("idx"))
        if requested and case_id not in requested:
            continue
        records.append(normalize_drb2_record(record))
        if limit is not None and len(records) >= limit:
            break
    found = {item["case_id"] for item in records}
    missing = requested - found
    if missing:
        raise ValueError("Unknown DRB II case IDs: " + ", ".join(sorted(missing)))
    if not records:
        raise ValueError("DRB II selection contains no cases")

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-drb2-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination
