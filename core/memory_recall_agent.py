from __future__ import annotations

import re

from .conversation_memory import ConversationMemory
from .models import MemoryRecallResult


class MemoryRecallAgent:
    """Findet gespeicherte Conversation-Memory-Fakten für direkte Nutzerfragen.

    MVP 19.1 bleibt bewusst regelbasiert: schnell, lokal, nachvollziehbar und ohne
    zusätzliche Abhängigkeiten. Spätere MVPs können hier semantische Suche ergänzen.
    """

    NAME_QUESTION_PATTERNS = [
        r"\bwie\s+heiße\s+ich\b",
        r"\bwie\s+heisse\s+ich\b",
        r"\bwas\s+ist\s+mein\s+name\b",
        r"\bkennst\s+du\s+meinen\s+namen\b",
        r"\bweißt\s+du\s+noch[,]?\s+wie\s+ich\s+heiße\b",
        r"\bweisst\s+du\s+noch[,]?\s+wie\s+ich\s+heisse\b",
        r"\bich\s+habe\s+meinen\s+namen\s+vergessen\b",
        r"\bich\s+hab\s+meinen\s+namen\s+vergessen\b",
    ]

    def __init__(self, memory: ConversationMemory | None = None):
        self.memory = memory or ConversationMemory()

    def recall(self, task: str) -> MemoryRecallResult:
        normalized = self._normalize(task)
        if not normalized:
            return MemoryRecallResult(
                recalled=False,
                confidence=0.0,
                reason="Empty input cannot be answered from memory.",
            )

        name_match = self._match_any(normalized, self.NAME_QUESTION_PATTERNS)
        if name_match:
            return self._answer_fact(
                key="name",
                answer_template="Du heißt {value}.",
                missing_answer="Deinen Namen habe ich noch nicht gespeichert.",
                matched_question=name_match,
                reason="User asks for their stored name.",
            )

        return MemoryRecallResult(
            recalled=False,
            confidence=0.0,
            reason="No matching memory recall pattern.",
        )

    def can_answer(self, task: str) -> bool:
        result = self.recall(task)
        return result.recalled and bool(result.answer)

    def _answer_fact(
        self,
        key: str,
        answer_template: str,
        missing_answer: str,
        matched_question: str,
        reason: str,
    ) -> MemoryRecallResult:
        facts = {fact.key: fact.value for fact in self.memory.facts()}
        value = facts.get(key)
        if value:
            return MemoryRecallResult(
                recalled=True,
                answer=answer_template.format(value=value),
                key=key,
                value=value,
                confidence=0.98,
                reason=reason,
                matched_question=matched_question,
            )

        return MemoryRecallResult(
            recalled=True,
            answer=missing_answer,
            key=key,
            confidence=0.72,
            reason=f"{reason} Fact is not stored yet.",
            matched_question=matched_question,
        )

    def _match_any(self, normalized: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return pattern
        return None

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())
