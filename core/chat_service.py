from __future__ import annotations

from .chat_response_router import ChatResponseRouter
from .chat_session_store import ChatSessionStore
from .conversation_memory import ConversationMemory
from .llm_chat_responder import LLMChatResponder
from .cognitive_context_builder import CognitiveContextBuilder
from .models import ChatRunResult
from .planner_worker_orchestrator import PlannerWorkerOrchestrator
from .tool_development_agent import ToolDevelopmentAgent
from .user_response import UserResponseFormatter


class ChatService:
    def __init__(self):
        self.store = ChatSessionStore()
        self.orchestrator = PlannerWorkerOrchestrator()
        self.formatter = UserResponseFormatter()
        self.router = ChatResponseRouter()
        self.chat_responder = LLMChatResponder()
        self.memory = ConversationMemory()
        self.knowledge_context = CognitiveContextBuilder()
        self.tool_development = ToolDevelopmentAgent()

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

        else:
            known_system_capability = self.router.is_known_system_capability(task)
            capability_gap = {}
            if not known_system_capability:
                capability_gap = self.tool_development.detect_gap(task, provider_name=provider_name, model=model)
            if (not known_system_capability) and (capability_gap.get("gap_detected") or capability_gap.get("safe_to_execute") is False):
                development = self.tool_development.analyze(
                    task,
                    auto_create=True,
                    provider_name=provider_name,
                    model=model,
                    precomputed_gap=capability_gap,
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
                    "error": development.error,
                }
                metadata = {"mode": "tool_development", "success": success, "proposal_id": proposal_id}

            elif (not known_system_capability) and self.router.should_use_tools(task):
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
                obsidian_diag = knowledge.get("obsidian", {}) or knowledge.get("diagnostics", {}).get("obsidian", {})
                if self._asks_for_vault(task) and obsidian_diag and not obsidian_diag.get("status_ok") and obsidian_diag.get("issues"):
                    issues = "; ".join(str(issue) for issue in obsidian_diag.get("issues", [])[:5])
                    answer = "Obsidian-Vault ist aktuell nicht verfügbar oder nicht konfiguriert: " + issues
                    success = True
                    plan = {}
                    execution = {
                        "success": True,
                        "final_output": {"message": answer},
                        "mode": "known_capability_preflight",
                        "provider_name": None,
                        "model": None,
                        "error": None,
                        "context_used": False,
                        "knowledge_context": knowledge.get("diagnostics", {}).get("knowledge_context", knowledge),
                    }
                    metadata = {"mode": "known_capability_preflight", "success": True, "knowledge_context": execution.get("knowledge_context", {})}
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

                if obsidian_diag.get("blocked_reason") and self._asks_for_vault(task):
                    answer = obsidian_diag.get("user_message") or f"Obsidian-Kontext wurde blockiert: {obsidian_diag.get('blocked_reason')}"
                    success = True
                    plan = {}
                    execution = {
                        "success": True,
                        "final_output": {"message": answer},
                        "mode": "cognitive_context_policy",
                        "provider_name": None,
                        "model": None,
                        "error": None,
                        "context_used": False,
                        "knowledge_context": knowledge.get("diagnostics", {}).get("knowledge_context", knowledge),
                    }
                    metadata = {"mode": "cognitive_context_policy", "success": True, "knowledge_context": execution.get("knowledge_context", {})}
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
    
                # Vault topic questions are factual index queries. Answer them directly
                # from Pandora's Obsidian index so the GUI chat does not depend on an
                # LLM guessing whether it has local file access. The LLM can still be
                # used for broader follow-up questions with the same context.
                if self._asks_for_vault_topics(task) and obsidian_diag.get("topics"):
                    answer = self._format_vault_topics_answer(obsidian_diag)
                    success = True
                    plan = {}
                    execution = {
                        "success": True,
                        "final_output": {"message": answer},
                        "mode": "cognitive_context_direct_answer",
                        "provider_name": None,
                        "model": None,
                        "error": None,
                        "context_used": True,
                        "knowledge_context": {
                            "source_count": knowledge.get("source_count", 0),
                            "sources": knowledge.get("sources", []),
                            "target": knowledge.get("target"),
                            "cloud_context": knowledge.get("cloud_context"),
                            "blocked_local_only_count": knowledge.get("blocked_local_only_count", 0),
                            "blocked_obsidian_count": knowledge.get("blocked_obsidian_count", 0),
                            "obsidian": obsidian_diag,
                            "route_target": knowledge.get("route_target", {}),
                            "cognitive_context": knowledge,
                        },
                    }
                    metadata = {"mode": "cognitive_context_direct_answer", "success": True, "context_used": True, "knowledge_context": execution.get("knowledge_context", {})}
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
                    "knowledge_context": {
                        "source_count": knowledge.get("source_count", 0),
                        "sources": knowledge.get("sources", []),
                        "target": knowledge.get("target"),
                        "cloud_context": knowledge.get("cloud_context"),
                        "blocked_local_only_count": knowledge.get("blocked_local_only_count", 0),
                        "blocked_obsidian_count": knowledge.get("blocked_obsidian_count", 0),
                        "obsidian": knowledge.get("obsidian", {}) or knowledge.get("diagnostics", {}).get("obsidian", {}),
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

    def _asks_for_vault(self, task: str) -> bool:
        q = (task or "").lower()
        return "vault" in q or "obsidian" in q

    def _asks_for_vault_topics(self, task: str) -> bool:
        q = (task or "").lower()
        return self._asks_for_vault(task) and any(word in q for word in [
            "topic", "topics", "themen", "thema", "tags", "schwerpunkte",
            "was steht", "was ist", "inhalt", "inhalte", "überblick", "ueberblick", "overview",
        ])

    def _format_vault_topics_answer(self, obsidian_diag: dict) -> str:
        topics = obsidian_diag.get("topics") or {}

        def fmt_pairs(values, prefix=""):
            items = []
            for value in values or []:
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    items.append(f"- {prefix}{value[0]} ({value[1]})")
                elif isinstance(value, dict):
                    name = value.get("name") or value.get("tag") or value.get("folder") or value.get("link")
                    count = value.get("count")
                    if name:
                        items.append(f"- {prefix}{name}" + (f" ({count})" if count is not None else ""))
            return items

        tag_lines = fmt_pairs(topics.get("tags"), prefix="#")
        link_lines = fmt_pairs(topics.get("wikilinks"), prefix="[[")
        link_lines = [line + "]]" if line.startswith("- [[") and not line.endswith(")") and not line.endswith("]]") else line for line in link_lines]
        # The previous comprehension cannot safely append closing brackets when a
        # count is present; build wikilinks explicitly for clean output.
        link_lines = []
        for item in topics.get("wikilinks") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                link_lines.append(f"- [[{item[0]}]] ({item[1]})")
        folder_lines = fmt_pairs(topics.get("folders"))

        lines = ["Ich habe deinen Obsidian-Vault über den Pandora-Connector ausgewertet. Erkannte Topics:", ""]
        lines.append("**Top-Tags**")
        lines.extend(tag_lines[:15] or ["- keine Tags gefunden"])
        lines.append("")
        lines.append("**Top-Wikilinks**")
        lines.extend(link_lines[:15] or ["- keine Wikilinks gefunden"])
        lines.append("")
        lines.append("**Top-Ordner**")
        lines.extend(folder_lines[:15] or ["- keine Ordner gefunden"])
        return "\n".join(lines)

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
