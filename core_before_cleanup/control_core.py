from __future__ import annotations

from typing import Any

from .coordinator_agent import CoordinatorAgent
from .core_status import CoreStatusService
from .memory_gateway import MemoryGateway
from .model_router import ModelRouter
from .safety_gate import SafetyGate


class ControlCore:
    """Pandora's stable switching center.

    This is the narrow waist of the system: UI/CLI/API can call it, and it
    delegates to existing agents without letting experimental growth code become
    part of the protected core contract.
    """

    def __init__(self):
        self.status_service = CoreStatusService()
        self.memory = MemoryGateway()
        self.safety = SafetyGate()
        self.router = ModelRouter()
        self.coordinator = CoordinatorAgent()

    def status(self) -> dict[str, Any]:
        return self.status_service.status()

    def routes(self) -> dict[str, Any]:
        return {"routes": self.router.all_routes()}

    def safety_check(self, action: str, paths: list[str] | None = None, approved: bool = False) -> dict[str, Any]:
        decision = self.safety.evaluate(action, paths=paths, approved=approved)
        return decision.model_dump()

    async def run(self, task: str, session_id: str | None = None, provider_name: str | None = None, model: str | None = None, save: bool = True) -> dict[str, Any]:
        self.memory.append_event("task_received", {"task": task, "session_id": session_id})
        result = await self.coordinator.run(task, session_id=session_id, provider_name=provider_name, model=model, save=save)
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
        self.memory.append_event("task_completed", {"success": payload.get("success"), "route": payload.get("route"), "error": payload.get("error")})
        return payload
