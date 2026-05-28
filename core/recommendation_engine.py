from __future__ import annotations


class RecommendationEngine:
    def recommend(self, rankings: dict, failures: dict, journal_entries: list[dict]) -> list[dict]:
        recommendations: list[dict] = []
        ranked = rankings.get("rankings", {})

        for key, stats in ranked.items():
            if stats.get("runs", 0) >= 2 and stats.get("success_rate", 0) >= 0.9:
                recommendations.append({
                    "type": "prefer_action",
                    "target": key,
                    "reason": "High observed success rate.",
                    "priority": "medium",
                })

            if stats.get("runs", 0) >= 2 and stats.get("success_rate", 1) < 0.5:
                recommendations.append({
                    "type": "review_action",
                    "target": key,
                    "reason": "Low observed success rate.",
                    "priority": "high",
                })

        for action, count in failures.get("by_action", {}).items():
            if count >= 2:
                recommendations.append({
                    "type": "investigate_failure_pattern",
                    "target": action,
                    "reason": f"{count} failures observed.",
                    "priority": "high",
                })

        if not recommendations:
            recommendations.append({
                "type": "collect_more_data",
                "target": "journal",
                "reason": "Not enough repeated usage patterns yet.",
                "priority": "low",
            })

        return recommendations
