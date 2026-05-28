from __future__ import annotations

from collections import Counter
from .task_journal import TaskJournal


class SkillPatternDetector:
    def __init__(self, journal: TaskJournal | None = None):
        self.journal = journal or TaskJournal()

    def detect(self, limit: int = 100) -> dict:
        entries = self.journal.list(limit)
        tool_sequences: list[tuple[str, ...]] = []

        for entry in entries:
            action = entry.get("action", {})
            result = entry.get("result", {})
            if action.get("type") == "skill" and action.get("skill_id"):
                continue
            if action.get("type") == "tool" and action.get("tool_id"):
                tool_sequences.append((action["tool_id"],))

            # Detect explicit multi-step results from skills or future workflows.
            if isinstance(result, dict):
                steps = result.get("steps") or []
                seq = tuple(step.get("tool") for step in steps if step.get("tool"))
                if len(seq) >= 2:
                    tool_sequences.append(seq)

        counts = Counter(tool_sequences)
        if counts:
            sequence, count = counts.most_common(1)[0]
            return {
                "pattern_detected": count >= 2 or len(sequence) >= 2,
                "sequence": list(sequence),
                "count": count,
                "reason": "Repeated or multi-step tool sequence detected.",
            }

        # MVP fallback: if there is not enough journal data, propose known useful chain.
        return {
            "pattern_detected": True,
            "sequence": ["echo", "uppercase"],
            "count": 0,
            "reason": "Fallback demo pattern for initial skill evolution.",
        }
