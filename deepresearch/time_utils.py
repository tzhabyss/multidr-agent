"""Project clock helpers with an explicit, configurable display timezone."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_TIMEZONE = "Asia/Shanghai"
TIMEZONE_ENV = "DEEPRESEARCH_TIMEZONE"


def timezone_name(configured: str | None = None) -> str:
    return (configured or os.getenv(TIMEZONE_ENV) or DEFAULT_TIMEZONE).strip()


def resolve_timezone(configured: str | None = None) -> tzinfo:
    name = timezone_name(configured)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        # Windows Python installations do not always ship the IANA timezone
        # database. Preserve the expected China offset without adding a hard
        # dependency on tzdata.
        if name == DEFAULT_TIMEZONE:
            return timezone(timedelta(hours=8), name=DEFAULT_TIMEZONE)
        raise ValueError(f"Unknown timezone: {name}") from None


def now(configured: str | None = None) -> datetime:
    return datetime.now(resolve_timezone(configured))


def now_iso(configured: str | None = None) -> str:
    return now(configured).isoformat()


def clock_time(configured: str | None = None) -> str:
    return now(configured).strftime("%H:%M:%S")


def convert_iso(value: str, configured: str | None = None) -> str:
    """Convert an existing ISO timestamp (including legacy UTC data)."""
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(resolve_timezone(configured)).isoformat(timespec="seconds")
