from __future__ import annotations

import re


class ChatResponseRouter:
    TOOL_HINTS = [
        "rechne",
        "berechne",
        "calculate",
        "csv",
        "datei",
        "uppercase",
        "gross",
        "groß",
        "reverse",
    ]

    def deterministic_existing_tool(self, task: str) -> str | None:
        """Return a known safe tool when no LLM/capability gate is needed.

        This is intentionally conservative. Capability discovery stays LLM-first
        for ambiguous or missing tools, but obvious local deterministic tool
        calls such as arithmetic should not spend two LLM calls before using the
        calculator.
        """
        text = task.strip().lower()
        if not text:
            return None

        if self._contains_arithmetic_expression(text):
            return "calculator"

        if "uppercase" in text or "groß" in text or "gross" in text:
            return "uppercase"

        if text.startswith("echo ") or text.startswith("wiederhole "):
            return "echo"

        return None

    def is_known_system_capability(self, task: str) -> bool:
        """Return True for built-in Pandora capabilities that must not be
        blocked by semantic tool-gap analysis.

        This is a conservative preflight for explicit system integrations, not
        capability discovery. It prevents the Semantic Capability Decision
        Engine from hijacking requests that Pandora can already route
        deterministically, such as Obsidian/Knowledge/Memory lookups.
        """
        text = task.strip().lower()
        if not text:
            return False
        if "obsidian" in text or "vault" in text:
            return True
        if any(marker in text for marker in [
            "knowledge base",
            "wissensbasis",
            "mein wissen",
            "memory",
            "erinnerung",
            "erinnerst du",
        ]):
            return True
        return False

    def should_use_tools(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False

        if self.deterministic_existing_tool(task):
            return True

        return any(hint in text for hint in self.TOOL_HINTS)

    def _contains_arithmetic_expression(self, text: str) -> bool:
        if re.search(r"\d+\s*[+\-*/]\s*\d+", text):
            return True
        if any(hint in text for hint in ["rechne", "berechne", "calculate"]):
            # Do not route meta/tool-creation requests to calculator merely
            # because they contain a number-related word.
            if "tool" in text or "werkzeug" in text or "fähigkeit" in text or "faehigkeit" in text:
                return False
            return bool(re.search(r"\d", text))
        return False
