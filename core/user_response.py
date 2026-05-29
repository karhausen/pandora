from __future__ import annotations


class UserResponseFormatter:
    GREETINGS = {"hallo", "hi", "hey", "guten morgen", "guten tag", "guten abend", "servus", "moin"}

    def format_answer(self, task: str, execution: dict) -> str:
        if not execution.get("success"):
            return execution.get("error") or "Die Aufgabe konnte nicht erfolgreich ausgeführt werden."

        output = execution.get("final_output")

        if isinstance(output, dict):
            message = str(output.get("message") or "")
            if self._is_technical_fallback(message):
                return self._friendly_fallback(task)

            if "result" in output:
                return str(output["result"])
            if "text" in output:
                return str(output["text"])
            if "message" in output:
                return str(output["message"])

        if output is None:
            return self._friendly_fallback(task)

        text = str(output)
        if self._is_technical_fallback(text):
            return self._friendly_fallback(task)
        return text

    def _friendly_fallback(self, task: str) -> str:
        normalized = task.strip().lower()
        if any(normalized.startswith(greeting) for greeting in self.GREETINGS):
            return "Hallo! Ich bin Pandora. Was möchtest du als Nächstes tun?"
        return "Ich habe verstanden. Dafür brauche ich aktuell kein spezielles Tool."

    def _is_technical_fallback(self, text: str) -> bool:
        return "no suitable tool or skill needed" in text.lower()
