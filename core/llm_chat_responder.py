from __future__ import annotations

from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType


class LLMChatResponder:
    def __init__(self):
        self.llm = LLMRuntime()

    def respond(
        self,
        task: str,
        history: list[dict] | None = None,
        context_summary: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> dict:
        if provider_name == "mock":
            return {
                "success": True,
                "answer": self._mock_answer(task, context_summary=context_summary),
                "provider_name": "mock",
                "model": model,
            }

        prompt = self._build_prompt(task, history or [], context_summary=context_summary)
        response = self.llm.complete(LLMRequest(
            task_type=LLMTaskType.CHAT,
            prompt=prompt,
            provider_name=provider_name,
            model=model,
            expect_json=False,
            timeout=30.0,
        ))

        routing_diagnostics = self._routing_diagnostics(response.raw)
        if response.success:
            answer = response.content.strip() or "Ich habe verstanden."
            if routing_diagnostics.get("fallback_used") and response.provider_name == "mock":
                # The mock client echoes the full prompt. For user-facing chat this is noisy;
                # keep the diagnostic details in execution metadata and return the friendly mock answer.
                answer = self._mock_answer(task, context_summary=context_summary)
            return {
                "success": True,
                "answer": answer,
                "provider_name": response.provider_name,
                "model": response.model,
                "routing_diagnostics": routing_diagnostics,
                "fallback_used": routing_diagnostics.get("fallback_used", False),
                "primary_provider_name": routing_diagnostics.get("primary_provider_name"),
                "primary_model": routing_diagnostics.get("primary_model"),
                "fallback_reason": routing_diagnostics.get("fallback_reason"),
            }

        return {
            "success": False,
            "answer": "Ich konnte gerade keine LLM-Antwort erzeugen.",
            "error": response.error,
            "provider_name": response.provider_name,
            "model": response.model,
            "routing_diagnostics": routing_diagnostics,
            "fallback_used": routing_diagnostics.get("fallback_used", False),
            "primary_provider_name": routing_diagnostics.get("primary_provider_name"),
            "primary_model": routing_diagnostics.get("primary_model"),
            "fallback_reason": routing_diagnostics.get("fallback_reason"),
        }

    def _build_prompt(self, task: str, history: list[dict], context_summary: str | None = None) -> str:
        last_messages = history[-10:]
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in last_messages
        )
        context_text = context_summary or ""
        context_instruction = (
            "Wenn im Abschnitt 'Gesprächskontext und bekannte Fakten' Vault-/Knowledge-Inhalte stehen, "
            "hast du Zugriff auf genau diese bereitgestellten Inhalte. Sage dann nicht, dass du keinen Zugriff hast. "
            "Beantworte die Frage anhand dieses Kontextes und nenne nach Möglichkeit sichtbare Quellen/Titel. "
            "Wenn kein passender Kontext vorhanden ist, sage klar, dass Pandora dazu nichts Relevantes gefunden hat."
            if context_text.strip()
            else "Es wurde kein Vault-/Knowledge-Kontext bereitgestellt. Beantworte nur allgemeine Fragen direkt."
        )
        return (
            "Du bist Pandora, ein lokaler hilfreicher KI-Agent. "
            "Antworte freundlich, kurz und praktisch. "
            "Nutze bekannte Fakten und den Gesprächsverlauf, wenn sie relevant sind. "
            f"{context_instruction}\n\n"
            f"Gesprächskontext und bekannte Fakten:\n{context_text}\n\n"
            f"Bisheriger Verlauf:\n{history_text}\n\n"
            f"Nutzer: {task}\n"
            "Pandora:"
        )

    def _mock_answer(self, task: str, context_summary: str | None = None) -> str:
        text = task.strip().lower()
        if any(q in text for q in ["wie heiße ich", "was ist mein name", "kennst du meinen namen"]):
            if context_summary and "name:" in context_summary.lower():
                for line in context_summary.splitlines():
                    if line.lower().startswith("- name:"):
                        return "Du heißt " + line.split(":", 1)[1].strip() + "."
            return "Deinen Namen habe ich noch nicht gespeichert."
        if any(text.startswith(g) for g in ["hallo", "hi", "hey", "guten morgen", "guten tag", "guten abend", "servus", "moin"]):
            return "Hallo! Ich bin Pandora. Was möchtest du als Nächstes tun?"
        if "was kannst" in text or "hilfe" in text:
            return "Ich kann Aufgaben planen, Tools ausführen, Skills nutzen, einfache Berechnungen erledigen und Gesprächskontext berücksichtigen."
        return "Ich habe dich verstanden. Ich nutze dabei den aktuellen Gesprächskontext."

    def _routing_diagnostics(self, raw) -> dict:
        if not isinstance(raw, dict):
            return {"fallback_used": False}
        trace = raw.get("pandora_routing_trace") or {}
        primary = trace.get("primary") or {}
        fallback = trace.get("fallback") or {}
        primary_error = trace.get("primary_error")
        fallback_used = bool(trace.get("fallback_used"))
        return {
            "decision": trace.get("decision") or ("fallback" if fallback_used else "primary"),
            "fallback_used": fallback_used,
            "requested_provider_name": trace.get("requested_provider_name"),
            "requested_model": trace.get("requested_model"),
            "primary_provider_name": primary.get("provider_name"),
            "primary_provider_type": primary.get("provider_type"),
            "primary_model": primary.get("model"),
            "primary_reason": primary.get("reason"),
            "primary_error": primary_error,
            "fallback_provider_name": fallback.get("provider_name"),
            "fallback_provider_type": fallback.get("provider_type"),
            "fallback_model": fallback.get("model"),
            "fallback_error": trace.get("fallback_error"),
            "fallback_reason": (
                f"Primary provider failed: {primary_error}"
                if fallback_used and primary_error
                else None
            ),
        }
