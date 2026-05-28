from __future__ import annotations

from collections import Counter


class FailureAnalyzer:
    def analyze(self, journal_entries: list[dict]) -> dict:
        failures = [entry for entry in journal_entries if not entry.get("success")]
        by_action = Counter()
        by_reason = Counter()

        for entry in failures:
            action = entry.get("action", {})
            action_key = action.get("tool_id") or action.get("skill_id") or action.get("type") or "unknown"
            by_action[action_key] += 1

            evaluation = entry.get("evaluation", {})
            reason = evaluation.get("reason") or entry.get("error") or "unknown"
            by_reason[str(reason)[:160]] += 1

        return {
            "total_failures": len(failures),
            "by_action": dict(by_action.most_common()),
            "by_reason": dict(by_reason.most_common()),
        }
