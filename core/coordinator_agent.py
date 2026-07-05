from __future__ import annotations

from datetime import datetime, UTC

from .chat_service import ChatService
from .conversation_memory import ConversationMemory
from .coordinator_log import CoordinatorLog
from .tool_development_agent import ToolDevelopmentAgent
from .capability_orchestrator import CapabilityOrchestrator
from .models import CoordinatorDecision, CoordinatorResult
from .model_router import ModelRouter


class CoordinatorAgent:
    def __init__(self):
        self.memory = ConversationMemory()
        self.chat_service = ChatService()
        self.tool_development = ToolDevelopmentAgent()
        self.capability_orchestrator = CapabilityOrchestrator()
        self.log = CoordinatorLog()
        self._last_tool_gap: dict | None = None

    def decide(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> CoordinatorDecision:
        """Select Pandora's route via LLM-led semantic capability orchestration.

        This method intentionally contains no keyword/pattern routing. The LLM
        receives Pandora's current capability snapshot and returns a structured
        recommendation. Python validates that recommendation before execution.
        """
        self._last_tool_gap = None
        decision_payload = self.capability_orchestrator.decide(
            task, provider_name=provider_name, model=model
        )
        if decision_payload["route"] == "tool_development":
            capability = decision_payload.get("missing_capability") or decision_payload.get("requested_tool") or "unknown_capability"
            self._last_tool_gap = {
                "analysis_available": True,
                "gap_detected": True,
                "safe_to_execute": False,
                "capability": capability,
                "reason": decision_payload.get("reason"),
                "semantic_decision": decision_payload,
            }
        return CoordinatorDecision(
            route=decision_payload["route"],
            reason=decision_payload.get("reason", "Semantic capability decision."),
            confidence=float(decision_payload.get("confidence") or 0.5),
            task=task,
            session_id=session_id,
            provider_name=provider_name,
            model=model,
        )

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        save: bool = True,
    ) -> CoordinatorResult:
        decision = self.decide(task, session_id=session_id, provider_name=provider_name, model=model)

        try:
            if decision.route == "tool_development":
                development = self.tool_development.analyze(
                    task,
                    auto_create=True,
                    provider_name=provider_name,
                    model=model,
                    precomputed_gap=self._last_tool_gap,
                )
                proposal = development.proposal or {}
                proposal_id = proposal.get("id")
                answer = development.message
                if proposal_id:
                    answer += f" Proposal-ID: {proposal_id}."

                result = CoordinatorResult(
                    success=development.error is None,
                    route=decision.route,
                    answer=answer,
                    decision=decision,
                    session_id=session_id,
                    plan={},
                    execution={
                        "success": development.error is None,
                        "mode": "tool_development",
                        "tool_development": development.model_dump(mode="json"),
                        "proposal_id": proposal_id,
                        "error": development.error,
                    },
                    error=development.error,
                )
            else:
                chat_result = await self.chat_service.run(
                    task,
                    session_id=session_id,
                    provider_name=provider_name,
                    model=model,
                    save=save,
                )

                result = CoordinatorResult(
                    success=chat_result.success,
                    route=decision.route,
                    answer=chat_result.answer,
                    decision=decision,
                    session_id=chat_result.session_id,
                    plan=chat_result.plan,
                    execution=chat_result.execution,
                    error=None,
                )

        except Exception as exc:
            result = CoordinatorResult(
                success=False,
                route=decision.route,
                answer="Die Aufgabe konnte nicht ausgeführt werden.",
                decision=decision,
                session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        if save:
            entry = result.model_dump(mode="json")
            entry["created_at"] = datetime.now(UTC).isoformat()
            self.log.append(entry)

        return result

    def logs(self, limit: int = 20) -> list[dict]:
        return self.log.list(limit)
