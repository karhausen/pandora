from __future__ import annotations


class ChatResponseRouter:
    """Legacy compatibility shim.

    The pre-30.0 router made routing decisions from request wording. That path
    is intentionally disabled. Current routing must go through
    CapabilityOrchestrator with a CapabilitySnapshot.
    """

    legacy_status = "disabled_use_capability_orchestrator"

    def deterministic_existing_tool(self, task: str) -> None:
        return None

    def should_use_tools(self, task: str) -> bool:
        return False

    def status(self) -> dict:
        return {
            "status": self.legacy_status,
            "replacement": "core.capability_orchestrator.CapabilityOrchestrator",
            "no_keyword_routing": True,
        }
