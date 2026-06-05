from __future__ import annotations

from .tool_registry import ToolRegistry


class CapabilityDetector:
    KEYWORDS = {
        "json_pretty": ["json format", "pretty json", "json hübsch", "json formatieren"],
        "text_reverse": ["reverse text", "text umdrehen", "rückwärts"],
        "word_count": [
            "word count",
            "count words",
            "wörter zählen",
            "woerter zaehlen",
            "wörter zaehlen",
            "wörter",
            "woerter",
            "wortanzahl",
            "anzahl der wörter",
            "anzahl wörter",
        ],
        "timestamp": ["timestamp", "zeitstempel"],
        "weather_lookup": [
            "aktuelles wetter",
            "aktuelle wetterdaten",
            "aktuelle wetterinformationen",
            "wetter abrufen",
            "wetterdaten abrufen",
            "wetterinformationen abrufen",
            "wetterbericht",
            "weather lookup",
            "current weather",
            "weather forecast",
        ],
    }

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()
        self.registry.discover()

    def _available_tool_id(self, capability: str) -> str | None:
        return self.registry.resolve_id(capability)

    def _existing_tool_ids(self) -> set[str]:
        return {tool.id for tool in self.registry.list()}

    def detect(self, task: str, analysis: dict | None = None) -> dict:
        task_l = task.lower()
        existing_tool_ids = self._existing_tool_ids()

        # If LLM already suggested a missing capability, prefer that.
        missing = []
        if analysis:
            missing = analysis.get("missing_capabilities") or []

        for capability in missing:
            available_tool = self._available_tool_id(capability)
            if not available_tool:
                return {
                    "gap_detected": True,
                    "capability": capability,
                    "reason": "LLM analysis reported missing capability.",
                    "existing_tools": sorted(existing_tool_ids),
                }

        for capability, keywords in self.KEYWORDS.items():
            if any(keyword in task_l for keyword in keywords):
                available_tool = self._available_tool_id(capability)
                if not available_tool:
                    return {
                        "gap_detected": True,
                        "capability": capability,
                        "reason": f"Task matched capability keywords for {capability}.",
                        "existing_tools": sorted(existing_tool_ids),
                    }
                return {
                    "gap_detected": False,
                    "capability": capability,
                    "reason": f"Capability {capability} is already covered by installed tool {available_tool}.",
                    "existing_tools": sorted(existing_tool_ids),
                    "tool_available": True,
                    "suggested_existing_tool": available_tool,
                }

        return {
            "gap_detected": False,
            "capability": None,
            "reason": "No missing capability detected.",
            "existing_tools": sorted(existing_tool_ids),
        }
