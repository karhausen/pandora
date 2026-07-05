from __future__ import annotations

from .chat_session_store import ChatSessionStore
from .conversation_memory import ConversationMemory
from .llm_chat_responder import LLMChatResponder
from .cognitive_context_builder import CognitiveContextBuilder
from .models import ChatRunResult
from .planner_worker_orchestrator import PlannerWorkerOrchestrator
from .tool_development_agent import ToolDevelopmentAgent
from .capability_orchestrator import CapabilityOrchestrator
from .knowledge_intent_router import KnowledgeIntentRouter
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
        self.knowledge_intent_router = KnowledgeIntentRouter()


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
        """Build policy-safe knowledge context for chat answers.

        The guard has two jobs:
        1. Do not load Vault/Knowledge for non-chat execution routes, so old
           project notes cannot bend a tool/task decision into another path.
        2. Do not blindly trust an LLM "answer_directly" recommendation when
           the local knowledge search can produce relevant, policy-approved
           context. This is a safety net for questions about stored user/project
           information, without routing by request keywords.
        """
        action = capability_decision.get("action")
        route = capability_decision.get("route")
        needed_sources = capability_decision.get("needed_sources") or []
        if route != "chat":
            return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "non_chat_route"}
        if action == "clarify":
            return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "clarify_does_not_need_knowledge"}

        if action in {"answer_with_context", "use_knowledge", "use_memory"} or needed_sources:
            payload = self.knowledge_context.build_for_chat(task, provider_name=provider_name, model=model)
            payload["guarded"] = False
            payload["guard_reason"] = "explicit_context_need"
            return payload

        if action == "answer_directly":
            # LLMs can be overconfident and say "answer_directly" for questions
            # whose answer actually lives in Vault/Memory. We run a bounded,
            # policy-aware retrieval and only attach context if the ranker found
            # relevant sources. This is not a route decision from user keywords;
            # it is retrieval validation against Pandora's knowledge stores.
            payload = self.knowledge_context.build_for_chat(task, provider_name=provider_name, model=model, limit=3)
            if payload.get("source_count", 0) > 0 and payload.get("context_text"):
                payload["guarded"] = False
                payload["guard_reason"] = "knowledge_safety_net_found_relevant_context"
                capability_decision["action"] = "answer_with_context"
                capability_decision["route"] = "chat"
                capability_decision["knowledge_safety_net"] = True
                return payload
            return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "answer_directly_no_relevant_knowledge_found", "diagnostics": payload.get("diagnostics", {})}

        return {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "no_explicit_context_need"}

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        save: bool = True,
    ) -> ChatRunResult:
        """Run MVP 30.4 chat flow: Knowledge/Vault + LLM only.

        This MVP intentionally does not execute tools, create proposals, or run
        planner/worker. Those layers come later after Vault/Memory and normal LLM
        interaction are stable.
        """
        if session_id:
            try:
                session = self.store.get(session_id)
            except FileNotFoundError:
                session = self.store.create(title=task[:60])
        else:
            session = self.store.create(title=task[:60])

        user_message = None
        if save:
            user_message = self.store.add_message(session.session_id, "user", task)
            self.memory.extract_and_store(task, session_id=session.session_id)

        current_session = self.store.get(session.session_id) if save else session
        history = [m.model_dump(mode="json") for m in current_session.messages]
        memory_context = self.memory.build_context(session.session_id, current_session.messages)

        knowledge_intent = self.knowledge_intent_router.decide(task, provider_name=provider_name, model=model)
        knowledge = {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "knowledge_not_needed"}
        if knowledge_intent.needs_knowledge:
            knowledge = self.knowledge_context.build_for_chat(task, provider_name=provider_name, model=model, limit=5)
            knowledge["guarded"] = False
            knowledge["guard_reason"] = "knowledge_intent_true"
        else:
            # MVP 30.4.2: Trust but verify. The LLM router may incorrectly
            # classify a personal/project knowledge question as direct chat.
            # We therefore run a bounded, policy-aware retrieval validation.
            # This is not keyword routing: the decision is based on actual
            # approved knowledge hits from Pandora's stores.
            probe = self.knowledge_context.build_for_chat(task, provider_name=provider_name, model=model, limit=3)
            if probe.get("source_count", 0) > 0 and probe.get("context_text"):
                knowledge = probe
                knowledge["guarded"] = False
                knowledge["guard_reason"] = "retrieval_validation_found_context"
                knowledge_intent.needs_knowledge = True
                knowledge_intent.reason = "Knowledge router said direct chat, but approved Vault/Knowledge retrieval found relevant context."
                knowledge_intent.mode = "retrieval_validated_knowledge"
            else:
                knowledge = {"source_count": 0, "sources": [], "context_text": "", "guarded": True, "guard_reason": "retrieval_validation_no_context", "diagnostics": probe.get("diagnostics", {})}

        merged_context = memory_context.summary
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

        capability_decision = {
            "mvp": "30.4",
            "route": "chat",
            "action": "answer_with_context" if knowledge_intent.needs_knowledge else "answer_directly",
            "knowledge_first": True,
            "tools_enabled": False,
            "planner_worker_enabled": False,
            "capability_gap_enabled": False,
            "reason": knowledge_intent.reason,
            "knowledge_intent": knowledge_intent.model_dump(),
            "no_keyword_routing": True,
        }
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
            "context_used": bool(merged_context),
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
            "context_used": bool(merged_context),
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
            plan={},
            execution=execution,
        )

    def create_session(self, title: str | None = None) -> dict:
        """Create a chat session for the GUI/API.

        Kept for API compatibility; MVP 30.4 only changed the chat run flow,
        not session management.
        """
        return self.store.create(title=title).model_dump(mode="json")

    def get_session(self, session_id: str) -> dict:
        """Return one chat session for the GUI/API."""
        return self.store.get(session_id).model_dump(mode="json")

    def list_sessions(self) -> list[dict]:
        """Return all chat sessions for the GUI/API."""
        return self.store.list()

    def delete_session(self, session_id: str) -> dict:
        """Delete one chat session for the GUI/API."""
        return self.store.delete(session_id)

