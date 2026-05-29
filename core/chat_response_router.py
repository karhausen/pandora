from __future__ import annotations

import re


class ChatResponseRouter:
    TOOL_HINTS = [
        "rechne",
        "berechne",
        "calculate",
        "csv",
        "datei",
        "tool",
        "uppercase",
        "gross",
        "groß",
        "word_count",
        "reverse",
    ]

    def should_use_tools(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False

        # Obvious arithmetic should still go to planner/worker.
        if re.search(r"\d+\s*[+\-*/]\s*\d+", text):
            return True

        return any(hint in text for hint in self.TOOL_HINTS)
