from __future__ import annotations

from .chat_session_store import ChatSessionStore
from .conversation_memory import ConversationMemory
from .llm_chat_responder import LLMChatResponder
from .cognitive_context_builder import CognitiveContextBuilder
from .models import ChatRunResult
from .user_response import UserResponseFormatter
from .llm_route_registry import LLMRoutePlanner, RouteRegistry


class ChatService:
    """Main chat service for MVP 30.4.

    The router does not decide from the user text. The LLM receives a prompt
    containing available routes and requests exactly one route. Python validates
    and dispatches that route. Tools, skill execution, planner/worker,
    capability-gap and tool-development paths are deliberately inactive here.
    """

    def __init__(self):
        self.store = ChatSessionStore()
        self.formatter = UserResponseFormatter()
        self.chat_responder = LLMChatResponder()
        self.memory = ConversationMemory()
        self.knowledge_context = CognitiveContextBuilder()
        self.route_registry = RouteRegistry()
        self.route_planner = LLMRoutePlanner()
        self._active_session_id: str | None = None

    def _build_final_context(self, route_result: dict, memory_summary: str) -> str:
        return self._build_final_context_from_results([route_result], memory_summary)

    def _build_final_context_from_results(self, route_results: list[dict], memory_summary: str) -> str:
        parts: list[str] = []
        if memory_summary:
            parts.append("Gesprächsgedächtnis:\n" + memory_summary)
        source_lines: list[str] = []
        for route_result in route_results:
            if route_result.get("context_text"):
                kind = route_result.get("kind") or route_result.get("route") or "context"
                parts.append(f"Von Pandora bereitgestellter Kontext ({kind}):\n" + route_result["context_text"])
            if route_result.get("sources"):
                for src in route_result.get("sources", []):
                    if isinstance(src, dict):
                        label = src.get("relative_path") or src.get("source_id") or src.get("source_type") or str(src)
                    else:
                        label = str(src)
                    source_lines.append(label)
        if source_lines:
            deduped = []
            seen = set()
            for label in source_lines:
                if label not in seen:
                    seen.add(label)
                    deduped.append(label)
            parts.append("Verwendbare Quellen:\n" + "\n".join(f"{idx}. {label}" for idx, label in enumerate(deduped, start=1)))
        return "\n\n".join(parts)

    def _respond_with_context(
        self,
        task: str,
        history: list[dict],
        context_text: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> dict:
        guarded_context = context_text
        if context_text:
            guarded_context = (
                "Pandora hat dir unten Kontext aus erlaubten Quellen bereitgestellt. "
                "Behaupte nicht, du hättest keinen Zugriff auf diesen Kontext. "
                "Wenn der Kontext nicht ausreicht, sage klar, was fehlt.\n\n"
                + context_text
            )
        return self.chat_responder.respond(
            task,
            history=history,
            context_summary=guarded_context,
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
    ) -> ChatRunResult:
        if session_id:
            try:
                session = self.store.get(session_id)
            except FileNotFoundError:
                session = self.store.create(title=task[:60])
        else:
            session = self.store.create(title=task[:60])

        self._active_session_id = session.session_id
        user_message = self.store.add_message(session.session_id, "user", task) if save else None

        if save:
            self.memory.extract_and_store(task, session_id=session.session_id)

        current_session = self.store.get(session.session_id) if save else session
        history = [m.model_dump(mode="json") for m in current_session.messages]
        memory_context = self.memory.build_context(session.session_id, current_session.messages)

        route_loop: list[dict] = []
        planner_loop: list[dict] = []
        seen_route_keys: set[tuple[str, str]] = set()
        max_route_rounds = 3

        for _round in range(max_route_rounds):
            planner_result = self.route_planner.choose_route(
                task,
                self.route_registry.available_specs(),
                history=history,
                provider_name=provider_name,
                model=model,
                route_results=route_loop,
            )
            planner_loop.append(self._planner_metadata(planner_result))
            route_request = planner_result["route_request"]
            route_key = (route_request.route, str(route_request.input.get("query") or route_request.input.get("question") or ""))

            if route_key in seen_route_keys and route_request.route not in {"direct_answer", "clarify_user"}:
                break
            seen_route_keys.add(route_key)

            route_result = self.route_registry.dispatch(
                route_request,
                task,
                self,
                provider_name=provider_name,
                model=model,
            )
            route_loop.append(route_result)

            if route_result.get("route") in {"direct_answer", "clarify_user"}:
                break

        last_route_result = route_loop[-1] if route_loop else {"route": "direct_answer", "context_text": "", "sources": [], "source_count": 0}

        if last_route_result.get("route") == "clarify_user":
            answer = last_route_result.get("question") or "Ich brauche dazu eine kurze Klärung."
            success = True
            plan = {}
            execution = {
                "success": True,
                "final_output": {"message": answer},
                "mode": "llm_led_route_registry",
                "route": "clarify_user",
                "route_planner": planner_loop[-1] if planner_loop else {},
                "route_loop": route_loop,
                "planner_loop": planner_loop,
                "error": None,
            }
            metadata = {
                "mode": "llm_led_route_registry",
                "success": True,
                "route": "clarify_user",
                "route_planner": planner_loop[-1] if planner_loop else {},
            }
        else:
            final_context = self._build_final_context_from_results(route_loop, memory_context.summary)
            llm_result = self._respond_with_context(
                task,
                history,
                final_context,
                provider_name=provider_name,
                model=model,
            )
            answer = llm_result.get("answer") or "Ich habe verstanden."
            success = bool(llm_result.get("success"))
            plan = {}
            execution = {
                "success": success,
                "final_output": {"message": answer},
                "mode": "llm_led_route_registry",
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
                "error": llm_result.get("error"),
                "fallback_used": llm_result.get("fallback_used", False),
                "primary_provider_name": llm_result.get("primary_provider_name"),
                "primary_model": llm_result.get("primary_model"),
                "fallback_reason": llm_result.get("fallback_reason"),
                "routing_diagnostics": llm_result.get("routing_diagnostics", {}),
                "route": last_route_result.get("route"),
                "route_planner": planner_loop[-1] if planner_loop else {},
                "route_loop": route_loop,
                "planner_loop": planner_loop,
                "context_used": bool(final_context),
                "available_routes": [r.id for r in self.route_registry.available_specs()],
                "disabled_future_routes": [r.id for r in self.route_registry.all_specs() if not r.enabled],
            }
            metadata = {
                "mode": "llm_led_route_registry",
                "success": success,
                "provider_name": llm_result.get("provider_name"),
                "model": llm_result.get("model"),
                "route": last_route_result.get("route"),
                "route_planner": planner_loop[-1] if planner_loop else {},
                "context_used": bool(final_context),
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

    def _planner_metadata(self, planner_result: dict) -> dict:
        request = planner_result.get("route_request")
        if request is None:
            request_payload = {}
        else:
            request_payload = {
                "route": request.route,
                "input": request.input,
                "reason": request.reason,
                "confidence": request.confidence,
            }
        return {
            "success": planner_result.get("success"),
            "provider_name": planner_result.get("provider_name"),
            "model": planner_result.get("model"),
            "error": planner_result.get("error"),
            "request": request_payload,
            "rule": "LLM chooses routes; Python validates and dispatches only.",
        }

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
