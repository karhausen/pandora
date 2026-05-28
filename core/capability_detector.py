from __future__ import annotations

from .tool_registry import ToolRegistry


class CapabilityDetector:
    KEYWORDS = {
        "json_pretty": ["json format", "pretty json", "json hübsch", "json formatieren"],
        "text_reverse": ["reverse text", "text umdrehen", "rückwärts"],
        "word_count": ["word count", "wörter zählen", "wortanzahl"],
        "timestamp": ["timestamp", "zeitstempel"],
    }

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()
        self.registry.discover()

    def detect(self, task: str, analysis: dict | None = None) -> dict:
        task_l = task.lower()
        existing_tool_ids = {tool.id for tool in self.registry.list()}

        # If LLM already suggested a missing capability, prefer that.
        missing = []
        if analysis:
            missing = analysis.get("missing_capabilities") or []

        for capability in missing:
            if capability not in existing_tool_ids:
                return {
                    "gap_detected": True,
                    "capability": capability,
                    "reason": "LLM analysis reported missing capability.",
                    "existing_tools": sorted(existing_tool_ids),
                }

        for capability, keywords in self.KEYWORDS.items():
            if any(keyword in task_l for keyword in keywords):
                if capability not in existing_tool_ids:
                    return {
                        "gap_detected": True,
                        "capability": capability,
                        "reason": f"Task matched capability keywords for {capability}.",
                        "existing_tools": sorted(existing_tool_ids),
                    }

        return {
            "gap_detected": False,
            "capability": None,
            "reason": "No missing capability detected.",
            "existing_tools": sorted(existing_tool_ids),
        }
