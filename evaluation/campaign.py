"""Persistent campaign-level token admission and accounting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_key(case_id: str, repetition: int) -> str:
    return f"{case_id}::r{repetition}"


class CampaignTokenLedger:
    """Reserve sample allowances and settle provider-reported usage.

    The ledger makes campaign admission deterministic and resumable. It stops
    new samples when the remaining budget cannot fund one full reservation.
    An in-flight provider call cannot be interrupted from this layer, so usage
    above its reservation is recorded as a breach and prevents further work.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_total_tokens: int,
        max_tokens_per_sample: int,
        resume: bool = False,
    ) -> None:
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if max_tokens_per_sample <= 0:
            raise ValueError("max_tokens_per_sample must be positive")
        if max_tokens_per_sample > max_total_tokens:
            raise ValueError("max_tokens_per_sample cannot exceed max_total_tokens")
        self.path = Path(path)
        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    "Campaign token ledger already exists; pass resume=True"
                )
            import json

            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            mismatches = []
            if int(self.data.get("max_total_tokens", 0)) != max_total_tokens:
                mismatches.append("max_total_tokens")
            if int(self.data.get("max_tokens_per_sample", 0)) != max_tokens_per_sample:
                mismatches.append("max_tokens_per_sample")
            if mismatches:
                raise ValueError(
                    "Campaign token configuration mismatch: " + ", ".join(mismatches)
                )
        else:
            self.data: dict[str, Any] = {
                "schema_version": "1.0",
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "max_total_tokens": max_total_tokens,
                "max_tokens_per_sample": max_tokens_per_sample,
                "status": "active",
                "stop_reason": "",
                "consumed_tokens": 0,
                "reserved_tokens": 0,
                "remaining_tokens": max_total_tokens,
                "entries": [],
            }
            self._persist()
        self._recompute()

    def _persist(self) -> None:
        self.data["updated_at"] = _utc_now()
        write_json(self.path, self.data)

    def _recompute(self) -> None:
        entries = list(self.data.get("entries", []))
        consumed = sum(
            int(item.get("consumed_tokens", 0))
            for item in entries
            if item.get("status") == "completed"
        )
        reserved = sum(
            int(item.get("reserved_tokens", 0))
            for item in entries
            if item.get("status") == "reserved"
        )
        maximum = int(self.data["max_total_tokens"])
        self.data["consumed_tokens"] = consumed
        self.data["reserved_tokens"] = reserved
        self.data["remaining_tokens"] = max(0, maximum - consumed - reserved)
        if consumed > maximum:
            self.data["status"] = "breached"
            self.data["stop_reason"] = "reported_usage_exceeded_campaign_budget"

    def reserve(self, case_id: str, repetition: int) -> int | None:
        key = _sample_key(case_id, repetition)
        for item in self.data["entries"]:
            if item.get("sample_key") != key:
                continue
            if item.get("status") == "reserved":
                return int(item["reserved_tokens"])
            if item.get("status") == "completed":
                return 0
        self._recompute()
        allowance = int(self.data["max_tokens_per_sample"])
        if self.data["status"] == "breached" or int(
            self.data["remaining_tokens"]
        ) < allowance:
            self.data["status"] = "exhausted"
            if not self.data.get("stop_reason"):
                self.data["stop_reason"] = "insufficient_tokens_for_next_sample"
            self._persist()
            return None
        self.data["entries"].append(
            {
                "sample_key": key,
                "case_id": case_id,
                "repetition": repetition,
                "status": "reserved",
                "reserved_tokens": allowance,
                "consumed_tokens": 0,
                "reserved_at": _utc_now(),
                "settled_at": None,
            }
        )
        self.data["status"] = "active"
        self.data["stop_reason"] = ""
        self._recompute()
        self._persist()
        return allowance

    def settle(
        self,
        case_id: str,
        repetition: int,
        *,
        consumed_tokens: int,
    ) -> None:
        if consumed_tokens < 0:
            raise ValueError("consumed_tokens cannot be negative")
        key = _sample_key(case_id, repetition)
        entry = next(
            (item for item in self.data["entries"] if item.get("sample_key") == key),
            None,
        )
        if entry is None:
            raise ValueError(f"Sample was not reserved: {key}")
        if entry.get("status") == "completed":
            if int(entry.get("consumed_tokens", 0)) != consumed_tokens:
                raise ValueError(f"Conflicting campaign settlement: {key}")
            return
        entry["status"] = "completed"
        entry["consumed_tokens"] = consumed_tokens
        entry["settled_at"] = _utc_now()
        entry["reservation_exceeded"] = consumed_tokens > int(
            entry.get("reserved_tokens", 0)
        )
        self._recompute()
        if entry["reservation_exceeded"]:
            self.data["status"] = "breached"
            self.data["stop_reason"] = "sample_reported_usage_exceeded_reservation"
        self._persist()

    def finalize(self) -> None:
        self._recompute()
        if self.data["status"] == "active":
            self.data["status"] = "completed"
        self._persist()
