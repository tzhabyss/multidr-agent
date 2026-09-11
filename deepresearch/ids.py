"""Stable identifiers and text normalization."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_hash(*parts: Any) -> str:
    normalized = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{stable_hash(*parts)[:24]}"

