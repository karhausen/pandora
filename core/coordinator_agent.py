from __future__ import annotations

from datetime import datetime, UTC

from .chat_response_router import ChatResponseRouter
from .chat_service import ChatService
from .conversation_memory import ConversationMemory
from .coordinator_log import CoordinatorLog
from .tool_development_agent import ToolDevelopmentAgent
from .models import CoordinatorDecision, CoordinatorResult


class CoordinatorAgent:
    def __init__(self):
        self.router = ChatResponseRouter()
        self.memory = ConversationMemory()
        self.chat_service = ChatService()
        self.tool_development = ToolDevelopmentAgent()
        self.log = CoordinatorLog()
        self._last_tool_gap: dict | None = None

    def decide(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = "mock",
        model: str | None = None,
    ) -> CoordinatorDecision:
        normalized = task.strip().lower()

        if self.memory.answer_from_memory(task):
            return CoordinatorDecision(
                route="memory",
                reason="Conversation memory can answer this directly.",
                confidence=0.95,
                task=task,
                session_id=session_id,
                provider_name=provider_name,
                model=model,
            )

        deterministic_tool = self.router.deterministic_existing_tool(task)
        if deterministic_tool:
            return CoordinatorDecision(
                route="planner_worker",
                reason=f"Known deterministic tool can handle this directly: {deterministic_tool}.",
                confidence=0.95,
                task=task,
                session_id=session_id,
                provider_name=provider_name,
                model=model,
            )

        self._last_tool_gap = None
        capability_gap = self.tool_development.detect_gap(task, provider_name=provider_name, model=model)
        if capability_gap.get("gap_detected"):
            self._last_tool_gap = capability_gap
            return CoordinatorDecision(
                route="tool_development",
                reason=capability_gap.get("reason", "Missing capability detected."),
                confidence=0.9,
                task=task,
                session_id=session_id,
                provider_name=provider_name,
                model=model,
            )

        if self.router.should_use_tools(task):
            return CoordinatorDecision(
                route="planner_worker",
                reason="Task appears to require a tool or structured execution.",
                confidence=0.85,
                task=task,
                session_id=session_id,
                provider_name=provider_name,
                model=model,
            )

        if normalized:
            return CoordinatorDecision(
                route="chat",
                reason="Task is conversational/free-form text.",
                confidence=0.8,
                task=task,
                session_id=session_id,
                provider_name=provider_name,
                model=model,
            )

        return CoordinatorDecision(
            route="chat",
            reason="Empty or unclear input; use chat fallback.",
            confidence=0.4,
            task=task,
            session_id=session_id,
            provider_name=provider_name,
            model=model,
        )

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = "mock",
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
