from __future__ import annotations

from collections import defaultdict


class ToolSkillRanker:
    def rank(self, journal_entries: list[dict]) -> dict:
        stats = defaultdict(lambda: {
            "runs": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
            "success_rate": 0.0,
            "avg_time": 0.0,
            "score": 0.0,
        })

        for entry in journal_entries:
            action = entry.get("action", {})
            action_type = action.get("type")
            key = None
            if action_type == "tool" and action.get("tool_id"):
                key = f"tool:{action['tool_id']}"
            elif action_type == "skill" and action.get("skill_id"):
                key = f"skill:{action['skill_id']}"

            if not key:
                continue

            s = stats[key]
            s["runs"] += 1
            if entry.get("success"):
                s["successes"] += 1
            else:
                s["failures"] += 1
            s["total_time"] += float(entry.get("execution_time") or 0.0)

        for key, s in stats.items():
            runs = max(1, s["runs"])
            s["success_rate"] = s["successes"] / runs
            s["avg_time"] = s["total_time"] / runs
            # Simple score: success dominates, speed is a small bonus.
            speed_bonus = max(0.0, 1.0 - min(s["avg_time"], 5.0) / 5.0) * 0.1
            s["score"] = round(s["success_rate"] + speed_bonus, 4)

        return {
            "rankings": dict(sorted(stats.items(), key=lambda item: item[1]["score"], reverse=True))
        }
