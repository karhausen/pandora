from __future__ import annotations

from .chat_response_router import ChatResponseRouter
from .chat_session_store import ChatSessionStore
from .conversation_memory import ConversationMemory
from .llm_chat_responder import LLMChatResponder
from .knowledge_context import KnowledgeContextService
from .models import ChatRunResult
from .planner_worker_orchestrator import PlannerWorkerOrchestrator
from .user_response import UserResponseFormatter


class ChatService:
    def __init__(self):
        self.store = ChatSessionStore()
        self.orchestrator = PlannerWorkerOrchestrator()
        self.formatter = UserResponseFormatter()
        self.router = ChatResponseRouter()
        self.chat_responder = LLMChatResponder()
        self.memory = ConversationMemory()
        self.knowledge_context = KnowledgeContextService()

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
        if memory_answer:
            answer = memory_answer
            success = True
            plan = {}
            execution = {
                "success": True,
                "final_output": {"message": answer},
                "mode": "conversation_memory",
                "error": None,
            }
            metadata = {"mode": "conversation_memory", "success": True}

        elif self.router.should_use_tools(task):
            result = await self.orchestrator.run(task, provider_name=provider_name, model=model, save=save)
            execution = result.get("execution", {})
            answer = self.formatter.format_answer(task, execution)
            plan = result.get("plan", {})
            success = bool(result.get("success"))
            metadata = {
                "mode": "planner_worker",
                "plan_id": plan.get("plan_id"),
                "execution_id": execution.get("execution_id"),
                "success": success,
            }

        else:
            current_session = self.store.get(session.session_id) if save else session
            history = [m.model_dump(mode="json") for m in current_session.messages]
            context = self.memory.build_context(session.session_id, current_session.messages)
            knowledge = self.knowledge_context.build_for_chat(task, provider_name=provider_name, model=model)
            merged_context = context.summary
            if knowledge.get("context_text"):
                merged_context = (merged_context + "\n\n" if merged_context else "") + "User Knowledge Base Kontext:\n" + knowledge["context_text"]
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
                "context_used": True,
                "knowledge_context": {
                    "source_count": knowledge.get("source_count", 0),
                    "sources": knowledge.get("sources", []),
                    "target": knowledge.get("target"),
                    "cloud_context": knowledge.get("cloud_context"),
                    "blocked_local_only_count": knowledge.get("blocked_local_only_count", 0),
                    "route_target": knowledge.get("route_target", {}),
                },
            }
            metadata = {
                "mode": "llm_chat",
                "success": success,
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
                "context_used": True,
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
