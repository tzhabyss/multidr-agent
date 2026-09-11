"""Small atomic JSON/JSONL helpers used by the evaluation runner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .schemas import EvaluationCase


class JsonlRecordConflict(ValueError):
    """Raised when one logical JSONL key has more than one payload."""


def _record_key(value: dict[str, Any], key_fields: tuple[str, ...]) -> tuple[Any, ...]:
    key = tuple(value.get(field) for field in key_fields)
    missing = [field for field, item in zip(key_fields, key) if item is None]
    if missing:
        raise ValueError("JSONL record is missing key field(s): " + ", ".join(missing))
    return key


def unique_records(
    values: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    record_name: str,
) -> list[dict[str, Any]]:
    """Collapse byte-equivalent logical rows and reject conflicting duplicates."""

    ordered: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for value in values:
        key = _record_key(value, key_fields)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = value
            ordered.append(value)
            continue
        if existing != value:
            rendered = ", ".join(
                f"{field}={item!r}" for field, item in zip(key_fields, key)
            )
            raise JsonlRecordConflict(
                f"Conflicting duplicate {record_name} record: {rendered}"
            )
    return ordered


def load_unique_jsonl(
    path: str | Path,
    *,
    key_fields: tuple[str, ...],
    record_name: str,
) -> list[dict[str, Any]]:
    return unique_records(
        load_jsonl(path), key_fields=key_fields, record_name=record_name
    )


def load_cases(path: str | Path) -> list[EvaluationCase]:
    path = Path(path)
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            case = EvaluationCase.model_validate(value)
        except Exception as exc:
            raise ValueError(f"Invalid JSONL case at line {line_number}: {exc}") from exc
        if case.case_id in seen:
            raise ValueError(f"Duplicate case_id at line {line_number}: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Dataset contains no cases")
    return cases


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL value must be an object at {path}:{line_number}")
        values.append(value)
    return values


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def append_unique_jsonl(
    path: str | Path,
    value: dict[str, Any],
    *,
    key_fields: tuple[str, ...],
    record_name: str,
) -> bool:
    """Append one logical row exactly once.

    Returns ``False`` for an identical retry and raises for a conflicting retry.
    This intentionally uses an atomic rewrite because evaluation files are small
    and correctness matters more than append throughput.
    """

    rows = load_unique_jsonl(
        path, key_fields=key_fields, record_name=record_name
    )
    key = _record_key(value, key_fields)
    for existing in rows:
        if _record_key(existing, key_fields) != key:
            continue
        if existing != value:
            rendered = ", ".join(
                f"{field}={item!r}" for field, item in zip(key_fields, key)
            )
            raise JsonlRecordConflict(
                f"Conflicting duplicate {record_name} record: {rendered}"
            )
        return False
    write_jsonl(path, [*rows, value])
    return True


def write_jsonl(path: str | Path, values: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file with the supplied records."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(value, ensure_ascii=False, default=str) + "\n"
        for value in values
    ).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
