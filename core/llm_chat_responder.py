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
        provider_name: str | None = "mock",
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

        if response.success:
            return {
                "success": True,
                "answer": response.content.strip() or "Ich habe verstanden.",
                "provider_name": response.provider_name,
                "model": response.model,
            }

        return {
            "success": False,
            "answer": "Ich konnte gerade keine LLM-Antwort erzeugen.",
            "error": response.error,
            "provider_name": response.provider_name,
            "model": response.model,
        }

    def _build_prompt(self, task: str, history: list[dict], context_summary: str | None = None) -> str:
        last_messages = history[-10:]
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in last_messages
        )
        context_text = context_summary or ""
        return (
            "Du bist Pandora, ein lokaler hilfreicher KI-Agent. "
            "Antworte freundlich, kurz und praktisch. "
            "Nutze bekannte Fakten und den Gesprächsverlauf, wenn sie relevant sind.\n\n"
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
