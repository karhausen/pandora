from __future__ import annotations

from .chat_session_store import ChatSessionStore
from .conversation_memory import ConversationMemory
from .llm_chat_responder import LLMChatResponder
from .cognitive_context_builder import CognitiveContextBuilder
from .models import ChatRunResult
from .planner_worker_orchestrator import PlannerWorkerOrchestrator
from .tool_development_agent import ToolDevelopmentAgent
from .capability_orchestrator import CapabilityOrchestrator
from .user_response import UserResponseFormatter


class ChatService:
    def __init__(self):
        self.store = ChatSessionStore()
        self.orchestrator = PlannerWorkerOrchestrator()
        self.formatter = UserResponseFormatter()
        self.chat_responder = LLMChatResponder()
        self.memory = ConversationMemory()
        self.knowledge_context = CognitiveContextBuilder()
        self.tool_development = ToolDevelopmentAgent()
        self.capability_orchestrator = CapabilityOrchestrator()


    def _clarification_answer(self, task: str, capability_decision: dict) -> str:
        missing = capability_decision.get("missing_capability") or capability_decision.get("requested_tool")
        if missing:
            return (
                f"Ich bin mir noch nicht sicher, ob dafür wirklich eine neue dauerhafte Pandora-Capability nötig ist. "
                f"Ich kann zuerst mit vorhandenen Fähigkeiten arbeiten, zum Beispiel per Python/Workflow. "
                f"Möchtest du {missing} nur einmal nutzen/berechnen lassen, oder soll ich daraus wirklich ein dauerhaftes Tool/Proposal für Pandora erstellen?"
            )
        return (
            "Ich brauche dazu eine kurze Klärung: Soll ich das mit vorhandenen Fähigkeiten lösen, "
            "oder möchtest du ausdrücklich eine neue dauerhafte Pandora-Capability als Proposal erstellen lassen?"
        )


    def _build_guarded_knowledge_context(
        self,
        task: str,
        capability_decision: dict,
        *,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Load knowledge only when the validated decision explicitly needs it.

        This prevents unrelated Vault notes or old test plans from overriding
        the active capability decision. The guard uses the structured decision,
        not keywords from the user request.
        """
        action = capability_decision.get("action")
        route = capability_decision.get("route")
        needed_sources = capability_decision.get("needed_sources") or []
        if route != "chat":
            return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "non_chat_route"}
        if action in {"answer_directly", "clarify"}:
            return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": f"{action}_does_not_need_knowledge"}
        if action not in {"answer_with_context", "use_knowledge", "use_memory"} and not needed_sources:
            return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "no_explicit_context_need"}
        return self.knowledge_context.build_for_chat(task, provider_name=provider_name, model=model)

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        save: bool = True,
    ) -> ChatRunResult:
        if session_id:
            try:
                session = self.store.get(session_id)
            except FileNotFoundError:
                session = self.store.create(title=task[:60])
        else:
            session = self.store.create(title=task[:60])

        user_message = self.store.add_message(session.session_id, "user", task) if save else None

        if save:
            self.memory.extract_and_store(task, session_id=session.session_id)

        memory_answer = self.memory.answer_from_memory(task)
        capability_decision = self.capability_orchestrator.decide(task, provider_name=provider_name, model=model)

        if capability_decision.get("action") == "clarify":
            answer = self._clarification_answer(task, capability_decision)
            success = True
            plan = {}
            execution = {
                "success": True,
                "final_output": {"message": answer},
                "mode": "clarification",
                "capability_decision": capability_decision,
                "error": None,
            }
            metadata = {"mode": "clarification", "success": True, "capability_decision": capability_decision}

        elif memory_answer and capability_decision.get("route") == "chat":
            answer = memory_answer
            success = True
            plan = {}
            execution = {
                "success": True,
                "final_output": {"message": answer},
                "mode": "conversation_memory",
                "capability_decision": capability_decision,
                "error": None,
            }
            metadata = {"mode": "conversation_memory", "success": True, "capability_decision": capability_decision}

        elif capability_decision.get("route") == "tool_development":
            capability = capability_decision.get("missing_capability") or capability_decision.get("requested_tool") or "unknown_capability"
            precomputed_gap = {
                "analysis_available": True,
                "gap_detected": True,
                "safe_to_execute": False,
                "capability": capability,
                "reason": capability_decision.get("reason"),
                "semantic_decision": capability_decision,
            }
            development = self.tool_development.analyze(
                task,
                auto_create=True,
                provider_name=provider_name,
                model=model,
                precomputed_gap=precomputed_gap,
            )
            proposal = development.proposal or {}
            proposal_id = proposal.get("id")
            answer = development.message + (f" Proposal-ID: {proposal_id}." if proposal_id else "")
            success = development.error is None
            plan = {}
            execution = {
                "success": success,
                "mode": "tool_development",
                "tool_development": development.model_dump(mode="json"),
                "proposal_id": proposal_id,
                "capability_decision": capability_decision,
                "error": development.error,
            }
            metadata = {"mode": "tool_development", "success": success, "proposal_id": proposal_id, "capability_decision": capability_decision}

        elif capability_decision.get("route") == "planner_worker":
            result = await self.orchestrator.run(task, provider_name=provider_name, model=model, save=save)
            execution = result.get("execution", {})
            answer = self.formatter.format_answer(task, execution)
            plan = result.get("plan", {})
            success = bool(result.get("success"))
            execution["capability_decision"] = capability_decision
            metadata = {
                "mode": "planner_worker",
                "plan_id": plan.get("plan_id"),
                "execution_id": execution.get("execution_id"),
                "success": success,
                "capability_decision": capability_decision,
            }

        else:
            current_session = self.store.get(session.session_id) if save else session
            history = [m.model_dump(mode="json") for m in current_session.messages]
            context = self.memory.build_context(session.session_id, current_session.messages)
            knowledge = self._build_guarded_knowledge_context(
                task,
                capability_decision,
                provider_name=provider_name,
                model=model,
            )
            merged_context = context.summary
            if knowledge.get("context_text"):
                merged_context = (merged_context + "\n\n" if merged_context else "") + "Knowledge Kontext (User Knowledge Base + freigegebene externe Quellen):\n" + knowledge["context_text"]
            llm_result = self.chat_responder.respond(
                task,
                history=history,
                context_summary=merged_context,
                provider_name=provider_name,
                model=model,
            )
            answer = llm_result.get("answer") or "Ich habe verstanden."
            success = bool(llm_result.get("success"))
            plan = {}
            execution = {
                "success": success,
                "final_output": {"message": answer},
                "mode": "llm_chat",
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
                "error": llm_result.get("error"),
                "fallback_used": llm_result.get("fallback_used", False),
                "primary_provider_name": llm_result.get("primary_provider_name"),
                "primary_model": llm_result.get("primary_model"),
                "fallback_reason": llm_result.get("fallback_reason"),
                "routing_diagnostics": llm_result.get("routing_diagnostics", {}),
                "context_used": True,
                "capability_decision": capability_decision,
                "knowledge_context": {
                    "source_count": knowledge.get("source_count", 0),
                    "sources": knowledge.get("sources", []),
                    "target": knowledge.get("target"),
                    "cloud_context": knowledge.get("cloud_context"),
                    "blocked_local_only_count": knowledge.get("blocked_local_only_count", 0),
                    "blocked_obsidian_count": knowledge.get("blocked_obsidian_count", 0),
                    "obsidian": knowledge.get("diagnostics", {}).get("obsidian", {}),
                    "route_target": knowledge.get("route_target", {}),
                    "cognitive_context": knowledge,
                },
            }
            metadata = {
                "mode": "llm_chat",
                "success": success,
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
                "fallback_used": llm_result.get("fallback_used", False),
                "primary_provider_name": llm_result.get("primary_provider_name"),
                "primary_model": llm_result.get("primary_model"),
                "fallback_reason": llm_result.get("fallback_reason"),
                "routing_diagnostics": llm_result.get("routing_diagnostics", {}),
                "context_used": True,
                "capability_decision": capability_decision,
                "knowledge_context": execution.get("knowledge_context", {}),
            }

        assistant_message = self.store.add_message(
            session.session_id,
            "assistant",
            answer,
            metadata=metadata,
        ) if save else None

        return ChatRunResult(
            session_id=session.session_id,
            success=success,
            answer=answer,
            user_message=user_message,
            assistant_message=assistant_message,
            plan=plan,
            execution=execution,
        )

    def create_session(self, title: str | None = None) -> dict:
        return self.store.create(title=title).model_dump(mode="json")

    def get_session(self, session_id: str) -> dict:
        return self.store.get(session_id).model_dump(mode="json")

    def list_sessions(self) -> list[dict]:
        return self.store.list()

    def delete_session(self, session_id: str) -> dict:
        return self.store.delete(session_id)

    def _answer_from_execution(self, execution: dict) -> str:
        return self.formatter.format_answer("", execution)
