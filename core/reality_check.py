from __future__ import annotations

import asyncio
from datetime import datetime, UTC

from .core_smoke_runner import CoreSmokeRunner
from .heartbeat import Heartbeat
from .models import RealityCheckIteration, RealityCheckResult
from .reality_check_log import RealityCheckLog
from .stability_reporter import StabilityReporter


class RealityCheck:
    def __init__(self):
        self.log = RealityCheckLog()
        self.reporter = StabilityReporter()

    async def run(self, iterations: int = 3, delay: float = 0.0, run_pytest: bool = False) -> RealityCheckResult:
        results: list[RealityCheckIteration] = []
        passed = 0

        for index in range(1, iterations + 1):
            heartbeat = await Heartbeat().check()
            smoke = await CoreSmokeRunner().run(run_pytest=run_pytest)

            success = bool(heartbeat.get("healthy")) and bool(smoke.success)
            if success:
                passed += 1

            item = RealityCheckIteration(
                iteration=index,
                heartbeat=heartbeat,
                smoke=smoke.model_dump(mode="json"),
                success=success,
            )
            results.append(item)

            if delay > 0 and index < iterations:
                await asyncio.sleep(delay)

        snapshot_summary = self.reporter.report()
        recommendations = self._recommend(results, snapshot_summary)

        final = RealityCheckResult(
            success=passed == iterations,
            iterations=iterations,
            passed=passed,
            failed=iterations - passed,
            results=results,
            snapshot_summary=snapshot_summary,
            recommendations=recommendations,
        )

        entry = final.model_dump(mode="json")
        entry["created_at"] = datetime.now(UTC).isoformat()
        self.log.append(entry)
        return final

    def logs(self, limit: int = 20) -> list[dict]:
        return self.log.list(limit)

    def report(self) -> dict:
        return self.reporter.report()

    def _recommend(self, results: list[RealityCheckIteration], summary: dict) -> list[str]:
        recs = []
        if any(not r.success for r in results):
            recs.append("At least one reality-check iteration failed. Review heartbeat/smoke details before adding new autonomy.")
        if summary.get("snapshots", {}).get("count", 0) > 20:
            recs.append("Many core snapshots exist. Consider pruning old FAILED/CANDIDATE snapshots.")
        if summary.get("memory", {}).get("total_size_bytes", 0) > 50_000_000:
            recs.append("Memory directory is growing. Consider archiving old logs.")
        if not recs:
            recs.append("System looks stable for the checked scope.")
        return recs
