from __future__ import annotations

from .tool_registry import ToolRegistry


class CapabilityDetector:
    """Compatibility detector after MVP 30.2 cleanup.

    This class no longer routes by request keywords. It only accepts structured
    LLM/capability-analysis output and checks whether the requested capability
    is already covered by the registry. Keep this class only for older callers;
    the current Coordinator path uses CapabilityOrchestrator.
    """

    legacy_status = "compatibility_only_no_keyword_routing"

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()
        self.registry.discover()

    def _available_tool_id(self, capability: str) -> str | None:
        return self.registry.resolve_id(capability)

    def _existing_tool_ids(self) -> set[str]:
        return {tool.id for tool in self.registry.list()}

    def detect(self, task: str, analysis: dict | None = None) -> dict:
        existing_tool_ids = sorted(self._existing_tool_ids())
        requested = []
        if analysis:
            for key in ("missing_capabilities", "needed_capabilities", "suggested_tools"):
                value = analysis.get(key) or []
                if isinstance(value, str):
                    value = [value]
                requested.extend(str(v) for v in value if v)

        for capability in requested:
            available_tool = self._available_tool_id(capability)
            if available_tool:
                return {
                    "gap_detected": False,
                    "capability": capability,
                    "reason": "Structured analysis requested a capability already covered by an installed tool.",
                    "existing_tools": existing_tool_ids,
                    "tool_available": True,
                    "suggested_existing_tool": available_tool,
                    "detector_mode": self.legacy_status,
                }
            return {
                "gap_detected": True,
                "capability": capability,
                "reason": "Structured analysis reported a missing capability.",
                "existing_tools": existing_tool_ids,
                "detector_mode": self.legacy_status,
            }

        return {
            "gap_detected": False,
            "capability": None,
            "reason": "No structured capability request was provided; no keyword fallback was used.",
            "existing_tools": existing_tool_ids,
            "detector_mode": self.legacy_status,
        }
