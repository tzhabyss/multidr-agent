"""Budget and pressure calculations shared by compaction stages."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal


PressureLevel = Literal[
    "normal", "light", "query_aware", "summarize", "emergency"
]


def estimate_tokens(value: str | Any) -> int:
    """Estimate mixed Chinese/English tokens without a model-specific tokenizer."""

    if not isinstance(value, str):
        value = str(value)
    if not value:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    non_cjk = len(value) - cjk
    mixed_estimate = cjk + math.ceil(non_cjk / 4)
    # Retain the old session compressor's conservative len/2 behavior so an
    # upgrade never delays compaction for long logs or minified tool output.
    return max(1, mixed_estimate, math.ceil(len(value) / 2))


@dataclass(frozen=True)
class CompactionPressure:
    level: PressureLevel
    estimated_tokens: int
    context_window_tokens: int
    ratio: float
    segment_multiplier: int


class CompactionController:
    """Translate a 100k-style working window into graduated pressure levels."""

    def __init__(
        self,
        *,
        context_window_tokens: int,
        light_ratio: float,
        query_ratio: float,
        summary_ratio: float,
        emergency_ratio: float,
    ) -> None:
        self.context_window_tokens = context_window_tokens
        self.light_ratio = light_ratio
        self.query_ratio = query_ratio
        self.summary_ratio = summary_ratio
        self.emergency_ratio = emergency_ratio

    def assess(self, estimated_tokens: int) -> CompactionPressure:
        ratio = estimated_tokens / max(1, self.context_window_tokens)
        if ratio >= self.emergency_ratio:
            level: PressureLevel = "emergency"
            multiplier = 4
        elif ratio >= self.summary_ratio:
            level = "summarize"
            multiplier = 3
        elif ratio >= self.query_ratio:
            level = "query_aware"
            multiplier = 2
        elif ratio >= self.light_ratio:
            level = "light"
            multiplier = 1
        else:
            level = "normal"
            multiplier = 1
        return CompactionPressure(
            level=level,
            estimated_tokens=estimated_tokens,
            context_window_tokens=self.context_window_tokens,
            ratio=ratio,
            segment_multiplier=multiplier,
        )

    @staticmethod
    def projection_level(
        raw_tokens: int,
        budget_tokens: int,
        *,
        light_ratio: float = 0.60,
        query_ratio: float = 0.75,
        summary_ratio: float = 0.88,
        emergency_ratio: float = 0.95,
    ) -> PressureLevel:
        ratio = raw_tokens / max(1, budget_tokens)
        if ratio >= emergency_ratio:
            return "emergency"
        if ratio >= summary_ratio:
            return "summarize"
        if ratio >= query_ratio:
            return "query_aware"
        if ratio >= light_ratio:
            return "light"
        return "normal"
