"""Visible benchmark date boundaries and conservative source-date admission."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import unquote, urlsplit


_MONTHS = {
    name.casefold(): index
    for index, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update(
    {
        name.casefold(): index
        for index, name in enumerate(calendar.month_abbr)
        if name
    }
)
_MONTH_PATTERN = "|".join(
    sorted((re.escape(item) for item in _MONTHS), key=len, reverse=True)
)


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _safe_date(year: int, month: int = 12, day: int | None = None) -> date | None:
    try:
        return date(year, month, day or calendar.monthrange(year, month)[1])
    except ValueError:
        return None


def extract_date_boundary(text: str) -> str | None:
    """Extract an inclusive cutoff from explicit English or Chinese wording."""

    value = str(text or "")
    cue = re.search(
        r"(?i)\b(?:up\s+to|through|as\s+of|until|by|before)\b|截至|截止(?:到|至)?",
        value,
    )
    if cue is None:
        return None
    tail = value[cue.end() : cue.end() + 80]
    exclusive = bool(re.search(r"(?i)\bbefore\b", cue.group(0)))

    chinese = re.search(
        r"(?P<year>20\d{2}|19\d{2})\s*年(?:\s*(?P<month>\d{1,2})\s*月)?"
        r"(?:\s*(?P<day>\d{1,2})\s*日)?",
        tail,
    )
    if chinese:
        year = int(chinese.group("year"))
        month = int(chinese.group("month") or 12)
        day = int(chinese.group("day")) if chinese.group("day") else None
        parsed = _safe_date(year, month, day)
        if parsed is not None:
            if exclusive:
                first_excluded = date(year, month, day or 1)
                parsed = date.fromordinal(first_excluded.toordinal() - 1)
            return parsed.isoformat()

    iso = re.search(
        r"(?P<year>20\d{2}|19\d{2})[-/](?P<month>\d{1,2})(?:[-/](?P<day>\d{1,2}))?",
        tail,
    )
    if iso:
        year = int(iso.group("year"))
        month = int(iso.group("month"))
        day = int(iso.group("day")) if iso.group("day") else None
        parsed = _safe_date(
            year,
            month,
            day,
        )
        if parsed is not None:
            if exclusive:
                first_excluded = date(year, month, day or 1)
                parsed = date.fromordinal(first_excluded.toordinal() - 1)
            return parsed.isoformat()

    month_first = re.search(
        rf"(?i)\b(?P<month>{_MONTH_PATTERN})\s+"
        r"(?:(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+))?"
        r"(?P<year>20\d{2}|19\d{2})\b",
        tail,
    )
    if month_first:
        year = int(month_first.group("year"))
        month = _MONTHS[month_first.group("month").casefold()]
        day = int(month_first.group("day")) if month_first.group("day") else None
        parsed = _safe_date(
            year,
            month,
            day,
        )
        if parsed is not None:
            if exclusive:
                first_excluded = date(year, month, day or 1)
                parsed = date.fromordinal(first_excluded.toordinal() - 1)
            return parsed.isoformat()

    year_only = re.search(r"\b(20\d{2}|19\d{2})\b", tail)
    if year_only:
        parsed = date(int(year_only.group(1)), 12, 31)
        if exclusive:
            parsed = date(parsed.year - 1, 12, 31)
        return parsed.isoformat()
    return None


def parse_date(value: Any) -> date | None:
    """Parse a provider publication timestamp without guessing from prose."""

    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    extracted = extract_date_boundary(text)
    if extracted:
        try:
            return date.fromisoformat(extracted)
        except ValueError:
            pass
    month_year = re.fullmatch(
        rf"(?i)\s*(?P<month>{_MONTH_PATTERN})\s+(?P<year>20\d{{2}}|19\d{{2}})\s*",
        text,
    )
    if month_year:
        return _month_end(
            int(month_year.group("year")),
            _MONTHS[month_year.group("month").casefold()],
        )
    return None


@dataclass(frozen=True)
class PublicationDate:
    value: date
    source: str


def infer_publication_date(
    hit: Any,
    *,
    final_url: str | None = None,
    final_title: str | None = None,
) -> PublicationDate | None:
    """Use only explicit metadata or strongly date-shaped URL/title signals."""

    explicit = parse_date(getattr(hit, "published_at", None))
    if explicit is not None:
        return PublicationDate(explicit, "search_metadata")

    url = unquote(str(final_url or getattr(hit, "url", "") or ""))
    path = urlsplit(url).path
    url_date = re.search(
        r"/(20\d{2}|19\d{2})/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])(?:/|$)",
        path,
    )
    if url_date:
        parsed = _safe_date(*(int(item) for item in url_date.groups()))
        if parsed is not None:
            return PublicationDate(parsed, "url_date")
    url_month = re.search(
        r"/(20\d{2}|19\d{2})/(0?[1-9]|1[0-2])(?:/|$)", path
    )
    if url_month:
        return PublicationDate(
            _month_end(int(url_month.group(1)), int(url_month.group(2))),
            "url_month",
        )

    metadata_text = " ".join(
        item
        for item in (
            str(final_title or ""),
            str(getattr(hit, "title", "") or ""),
            str(getattr(hit, "snippet", "") or "")[:500],
        )
        if item
    )
    labelled = re.search(
        r"(?i)\b(?:published|updated|publication\s+date|date)\s*[:\-]\s*"
        r"(?P<date>20\d{2}[-/]\d{1,2}[-/]\d{1,2}|"
        rf"(?:{_MONTH_PATTERN})\s+\d{{1,2}},\s*\d{{4}})",
        metadata_text,
    )
    if labelled:
        parsed = parse_date(labelled.group("date"))
        if parsed is not None:
            return PublicationDate(parsed, "labelled_metadata")
    return None


def source_within_date_boundary(
    hit: Any,
    boundary: str | None,
    *,
    final_url: str | None = None,
    final_title: str | None = None,
) -> tuple[bool, str | None]:
    cutoff = parse_date(boundary)
    if cutoff is None:
        return True, None
    publication = infer_publication_date(
        hit, final_url=final_url, final_title=final_title
    )
    if publication is None:
        return True, None
    return publication.value <= cutoff, publication.value.isoformat()
