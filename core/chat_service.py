from __future__ import annotations

from .chat_session_store import ChatSessionStore
from .models import ChatRunResult
from .planner_worker_orchestrator import PlannerWorkerOrchestrator
from .chat_response_router import ChatResponseRouter
from .llm_chat_responder import LLMChatResponder
from .user_response import UserResponseFormatter


class ChatService:
    def __init__(self):
        self.store = ChatSessionStore()
        self.orchestrator = PlannerWorkerOrchestrator()
        self.formatter = UserResponseFormatter()
        self.router = ChatResponseRouter()
        self.chat_responder = LLMChatResponder()

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        provider_name: str | None = "mock",
        model: str | None = None,
        save: bool = True,
    ) -> ChatRunResult:
        if session_id:
            session = self.store.get(session_id)
        else:
            session = self.store.create(title=task[:60])

        user_message = self.store.add_message(session.session_id, "user", task) if save else None

        if self.router.should_use_tools(task):
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
            llm_result = self.chat_responder.respond(task, history=history, provider_name=provider_name, model=model)
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
            }
            metadata = {
                "mode": "llm_chat",
                "success": success,
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
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
        if not execution.get("success"):
            return execution.get("error") or "Die Aufgabe konnte nicht erfolgreich ausgeführt werden."
        output = execution.get("final_output")
        if isinstance(output, dict):
            if "result" in output:
                return str(output["result"])
            if "text" in output:
                return str(output["text"])
            if "message" in output:
                return str(output["message"])
        if output is None:
            return "Erledigt."
        return str(output)
