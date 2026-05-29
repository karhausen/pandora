from __future__ import annotations

import json
import re
from datetime import datetime, UTC
from pathlib import Path

from .config import CONVERSATION_MEMORY_FILE
from .conversation_memory_log import ConversationMemoryLog
from .models import ChatMessage, ConversationContext, ConversationMemoryFact


class ConversationMemory:
    def __init__(self, path: Path = CONVERSATION_MEMORY_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = ConversationMemoryLog()

    def load(self) -> dict:
        if not self.path.exists():
            return {"facts": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def remember_fact(self, key: str, value: str, session_id: str | None = None) -> ConversationMemoryFact:
        now = datetime.now(UTC).isoformat()
        data = self.load()
        facts = data.setdefault("facts", {})
        existing = facts.get(key, {})
        fact = ConversationMemoryFact(
            key=key,
            value=value,
            source_session_id=session_id or existing.get("source_session_id"),
            created_at=existing.get("created_at") or now,
            updated_at=now,
        )
        facts[key] = fact.model_dump(mode="json")
        self.save(data)
        self.log.append({"event": "remember_fact", "fact": fact.model_dump(mode="json")})
        return fact

    def forget_fact(self, key: str) -> dict:
        data = self.load()
        existed = key in data.get("facts", {})
        data.get("facts", {}).pop(key, None)
        self.save(data)
        event = {"event": "forget_fact", "key": key, "existed": existed}
        self.log.append(event)
        return event

    def facts(self) -> list[ConversationMemoryFact]:
        data = self.load()
        return [ConversationMemoryFact.model_validate(v) for v in data.get("facts", {}).values()]

    def build_context(self, session_id: str, messages: list[ChatMessage], limit: int = 10) -> ConversationContext:
        recent = messages[-limit:]
        facts = self.facts()
        summary_lines = []
        if facts:
            summary_lines.append("Bekannte Fakten:")
            for fact in facts:
                summary_lines.append(f"- {fact.key}: {fact.value}")
        if recent:
            summary_lines.append("Letzte Nachrichten:")
            for message in recent:
                summary_lines.append(f"- {message.role}: {message.content}")
        return ConversationContext(
            session_id=session_id,
            recent_messages=recent,
            facts=facts,
            summary="\n".join(summary_lines),
        )

    def extract_and_store(self, text: str, session_id: str | None = None) -> list[ConversationMemoryFact]:
        facts: list[ConversationMemoryFact] = []
        normalized = text.strip()

        if re.search(r"vergiss (meinen namen|wie ich heiße|name)", normalized, re.IGNORECASE):
            self.forget_fact("name")
            return facts

        patterns = [
            (r"\bich heiße\s+([A-Za-zÄÖÜäöüß\- ]{2,60})", "name"),
            (r"\bmein name ist\s+([A-Za-zÄÖÜäöüß\- ]{2,60})", "name"),
        ]

        for pattern, key in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                value = self._clean_value(match.group(1))
                if value:
                    facts.append(self.remember_fact(key, value, session_id=session_id))
        return facts

    def answer_from_memory(self, text: str) -> str | None:
        normalized = text.strip().lower()
        facts = {fact.key: fact.value for fact in self.facts()}
        if any(q in normalized for q in ["wie heiße ich", "was ist mein name", "kennst du meinen namen"]):
            if "name" in facts:
                return f"Du heißt {facts['name']}."
            return "Deinen Namen habe ich noch nicht gespeichert."
        return None

    def _clean_value(self, value: str) -> str:
        value = value.strip().strip(".!?,;:")
        value = re.split(r"\s+(und|aber|weil|denn)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0]
        return value.strip()
