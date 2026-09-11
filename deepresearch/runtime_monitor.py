"""Periodic, bounded operational summaries with no routing side effects."""
import asyncio
import math
from collections import defaultdict


class RuntimeMonitor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.task = None
        self.latest = {}

    def summary(self, rows=None):
        groups = defaultdict(list)
        for row in list(self.runtime.records) if rows is None else rows:
            groups[(row["kind"], row["name"], row.get("version", ""),
                    row.get("run_id", ""), row.get("profile_id", ""),
                    row.get("provider_id", ""))].append(row)
        result = {}
        for (kind, name, version, run_id, profile_id, provider_id), values in groups.items():
            attempts = [r for r in values if r["cache_status"] != "hit" and r["error_category"] not in
                        {"cancelled", "authorization", "validation", "circuit_open"}]
            latencies = sorted(r["latency_ms"] for r in attempts)
            failures = 0
            for r in reversed(attempts):
                if r["error_category"] not in {"timeout", "connection", "rate_limit", "service"}:
                    break
                failures += 1
            key = f"{kind}:{name}:{version}"
            if run_id or profile_id or provider_id:
                key += f":{run_id}:{profile_id}:{provider_id}"
            result[key] = {
                "run_id": run_id, "profile_id": profile_id, "provider_id": provider_id,
                "total": len(values), "attempts": len(attempts),
                "success_rate": sum(r["status"] in {"success", "empty"} for r in attempts) / len(attempts) if attempts else None,
                "empty_results": sum(r["status"] == "empty" for r in attempts),
                "cache_hits": sum(r["cache_status"] == "hit" for r in values),
                "avg_ms": sum(latencies) / len(latencies) if latencies else None,
                "p95_ms": latencies[math.ceil(len(latencies) * .95) - 1] if len(latencies) >= 20 else None,
                "consecutive_failures": failures,
                "circuit_state": values[-1]["circuit_state"],
            }
        return {"metrics": result, "alerts": [key for key, value in result.items()
                if value["consecutive_failures"] >= 3 or value["circuit_state"] != "closed"],
                "routing_mode": "off"}

    async def start(self):
        if self.task is None and self.runtime.config.enabled:
            self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        while True:
            self.latest = self.summary()
            await asyncio.sleep(self.runtime.config.monitor_interval_seconds)

    async def close(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        self.latest = self.summary()
