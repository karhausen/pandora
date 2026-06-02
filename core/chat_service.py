from __future__ import annotations

from .chat_response_router import ChatResponseRouter
from .chat_session_store import ChatSessionStore
from .conversation_memory import ConversationMemory
from .llm_chat_responder import LLMChatResponder
from .memory_recall_agent import MemoryRecallAgent
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
        self.memory_recall = MemoryRecallAgent(self.memory)

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = "mock",
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

        memory_recall = self.memory_recall.recall(task)
        if memory_recall.recalled and memory_recall.answer:
            answer = memory_recall.answer
            success = True
            plan = {}
            execution = {
                "success": True,
                "final_output": {"message": answer},
                "mode": "memory_recall",
                "recall": memory_recall.model_dump(mode="json"),
                "error": None,
            }
            metadata = {
                "mode": "memory_recall",
                "success": True,
                "recall": memory_recall.model_dump(mode="json"),
            }

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
            llm_result = self.chat_responder.respond(
                task,
                history=history,
                context_summary=context.summary,
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
            }
            metadata = {
                "mode": "llm_chat",
                "success": success,
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
                "context_used": True,
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
