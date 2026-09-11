"""Dependency-free lexical and hashed-vector retrieval helpers."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from datetime import datetime


def normalize_content(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def content_hash(value: str) -> str:
    return hashlib.sha256(normalize_content(value).encode("utf-8")).hexdigest()


def terms(value: str) -> set[str]:
    normalized = normalize_content(value)
    result = set(re.findall(r"[a-z0-9][a-z0-9_.-]{1,}|[\u4e00-\u9fff]{2,}", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    result.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    return {item for item in result if item}


def lexical_index_text(*values: str) -> str:
    """Build one deterministic FTS representation for indexing and querying.

    SQLite's default ``unicode61`` tokenizer does not segment continuous
    Chinese text into the bigrams produced by :func:`terms`.  Materializing the
    features as whitespace-separated tokens makes the two sides symmetric
    while retaining Latin identifiers, numbers and long Chinese spans.
    """

    features: set[str] = set()
    for value in values:
        features.update(terms(value))
    return " ".join(sorted(features, key=lambda item: (item.casefold(), item)))


def fts_query_expression(value: str, *, maximum_terms: int = 20) -> str:
    """Return a safely quoted OR expression over the v2 lexical features."""

    selected = sorted(terms(value), key=lambda item: (-len(item), item))[:maximum_terms]
    return " OR ".join(
        f'"{item.replace(chr(34), chr(34) * 2)}"' for item in selected
    )


def hashed_embedding(value: str, dimensions: int = 128) -> list[float]:
    """Create a stable local vector without a hosted embedding dependency.

    Word/bigram features make this a useful semantic-ish fallback.  The model
    identifier is versioned by callers so a real embedding provider can replace
    it later through the same index table.
    """

    vector = [0.0] * dimensions
    features = list(terms(value))
    normalized = normalize_content(value)
    features.extend(normalized[index : index + 3] for index in range(max(0, len(normalized) - 2)))
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % dimensions
        vector[index] += -1.0 if number & 1 else 1.0
    norm = math.sqrt(sum(item * item for item in vector))
    return [item / norm for item in vector] if norm else vector


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(value: bytes, dimensions: int) -> list[float]:
    if len(value) != dimensions * 4:
        return []
    return list(struct.unpack(f"<{dimensions}f", value))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


def lexical_overlap(query: str, content: str) -> float:
    query_terms = terms(query)
    if not query_terms:
        return 0.0
    content_terms = terms(content)
    return min(1.0, len(query_terms & content_terms) / max(1, len(query_terms)))


def relevance_gate_score(
    *,
    semantic: float,
    lexical: float,
    semantic_margin: float = 0.0,
    channel_agreement: bool = False,
) -> float:
    """Score whether a top candidate is relevant, independently of fusion rank.

    RRF scores change scale when the number of available channels changes, so
    they are unsuitable as an abstention threshold.  This score remains on a
    0..1 relevance scale and uses only bounded evidence available at runtime.
    The final threshold is calibrated on a frozen development set.
    """

    semantic = max(0.0, min(1.0, semantic))
    lexical = max(0.0, min(1.0, lexical))
    margin_signal = max(0.0, min(1.0, semantic_margin * 5.0))
    evidence = max(semantic, lexical)
    supporting = min(semantic, lexical)
    return min(
        1.0,
        evidence
        + 0.03 * supporting
        + 0.04 * float(channel_agreement)
        + 0.02 * margin_signal,
    )


def recency_score(updated_at: str, half_life_days: float = 90.0) -> float:
    try:
        then = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        now = datetime.now(then.tzinfo)
        days = max(0.0, (now - then).total_seconds() / 86400)
    except (TypeError, ValueError):
        return 0.5
    return math.exp(-math.log(2) * days / half_life_days)
