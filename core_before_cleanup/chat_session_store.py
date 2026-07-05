from __future__ import annotations

import json
import uuid
from datetime import datetime, UTC
from pathlib import Path

from .config import CHAT_SESSION_INDEX_FILE, CHAT_SESSIONS_DIR
from .models import ChatMessage, ChatSession


class ChatSessionStore:
    def __init__(self, root: Path = CHAT_SESSIONS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        CHAT_SESSION_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    def create(self, title: str | None = None) -> ChatSession:
        now = datetime.now(UTC).isoformat()
        session = ChatSession(
            session_id=f"chat_{uuid.uuid4().hex[:12]}",
            title=title or "Neue Unterhaltung",
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self.save(session)
        self._update_index(session)
        return session

    def get(self, session_id: str) -> ChatSession:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(session_id)
        return ChatSession.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, session: ChatSession) -> None:
        self._path(session.session_id).write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._update_index(session)

    def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> ChatMessage:
        session = self.get(session_id)
        now = datetime.now(UTC).isoformat()
        message = ChatMessage(role=role, content=content, created_at=now, metadata=metadata or {})
        session.messages.append(message)
        session.updated_at = now
        if session.title == "Neue Unterhaltung" and role == "user":
            session.title = content[:60]
        self.save(session)
        return message

    def list(self) -> list[dict]:
        if not CHAT_SESSION_INDEX_FILE.exists():
            return []
        data = json.loads(CHAT_SESSION_INDEX_FILE.read_text(encoding="utf-8"))
        return sorted(data.get("sessions", []), key=lambda x: x.get("updated_at", ""), reverse=True)

    def delete(self, session_id: str) -> dict:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
        sessions = [s for s in self.list() if s.get("session_id") != session_id]
        CHAT_SESSION_INDEX_FILE.write_text(json.dumps({"sessions": sessions}, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"deleted": True, "session_id": session_id}

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _update_index(self, session: ChatSession) -> None:
        sessions = [s for s in self.list() if s.get("session_id") != session.session_id]
        sessions.append({
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": len(session.messages),
        })
        CHAT_SESSION_INDEX_FILE.write_text(json.dumps({"sessions": sessions}, indent=2, ensure_ascii=False), encoding="utf-8")
