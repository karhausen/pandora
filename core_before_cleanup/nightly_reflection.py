from __future__ import annotations

from collections import Counter
from datetime import datetime, UTC
from typing import Any

from .task_journal import TaskJournal
from .tool_lifecycle_manager import ToolLifecycleManager
from .memory_gateway import MemoryGateway


class NightlyReflection:
    """Offline learning pass: observe, summarize, propose. No auto-install."""

    def __init__(self):
        self.journal = TaskJournal()
        self.memory = MemoryGateway()

    def run(self, limit: int = 200) -> dict[str, Any]:
        entries = self.journal.list(limit)
        routes = Counter(str(e.get("route") or e.get("result", {}).get("route") or "unknown") for e in entries)
        failures = [e for e in entries if e.get("success") is False or e.get("error")]
        recommendations = []
        if failures:
            recommendations.append("Fehlerfälle clustern und gezielte Regressionstests daraus erzeugen.")
        if routes.get("tool_development", 0) > 0:
            recommendations.append("Wiederholte Fähigkeitslücken als Skill- oder Tool-Kandidaten prüfen.")
        recommendations.append("Keine Änderung automatisch aktivieren; Review-Paket für den User erzeugen.")

        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "mode": "nightly_reflection",
            "entries_analyzed": len(entries),
            "route_counts": dict(routes),
            "failure_count": len(failures),
            "recommendations": recommendations,
            "auto_changes_made": False,
        }
        self.memory.append_event("nightly_reflection", report)
        return report
