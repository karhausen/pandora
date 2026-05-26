from __future__ import annotations

from .models import CapabilityAnalysis
from .tool_registry import ToolRegistry


KEYWORD_CAPABILITIES = {
    "csv": "csv_processing",
    "excel": "spreadsheet_processing",
    "xlsx": "spreadsheet_processing",
    "json": "json_processing",
    "datei": "file_processing",
    "file": "file_processing",
    "calculate": "calculation",
    "rechne": "calculation",
    "summe": "calculation",
    "average": "calculation",
    "durchschnitt": "calculation",
}


class CapabilityAnalyzer:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def analyze(self, task: str) -> CapabilityAnalysis:
        task_l = task.lower()
        required = sorted({cap for key, cap in KEYWORD_CAPABILITIES.items() if key in task_l})
        available_tools = [tool.id for tool in self.registry.list()]
        available_caps = set()
        for tool in self.registry.list():
            text = f"{tool.name} {tool.description} {tool.id}".lower()
            if "csv" in text:
                available_caps.add("csv_processing")
            if "calculator" in text or "calculate" in text or "rechner" in text:
                available_caps.add("calculation")
            if "json" in text:
                available_caps.add("json_processing")
            if "file" in text or "datei" in text:
                available_caps.add("file_processing")

        missing = [cap for cap in required if cap not in available_caps]
        action = "create_tool" if missing else "direct_or_tool"
        return CapabilityAnalysis(
            task=task,
            required_capabilities=required,
            available_tools=available_tools,
            missing_capabilities=missing,
            recommended_action=action,
        )
