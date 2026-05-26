from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, UTC
from pathlib import Path
from .config import EPISODIC_DB
from .models import Episode

class EpisodicMemory:
    def __init__(self, path: Path = EPISODIC_DB):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    def _connect(self):
        return sqlite3.connect(self.path)
    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, task TEXT NOT NULL, kind TEXT NOT NULL, success INTEGER NOT NULL, used_tools TEXT NOT NULL, used_skills TEXT NOT NULL, execution_time REAL NOT NULL, error TEXT, summary TEXT, created_at TEXT NOT NULL)''')
    def record(self, task: str, kind: str, success: bool, used_tools: list[str] | None = None, used_skills: list[str] | None = None, execution_time: float = 0.0, error: str | None = None, summary: str | None = None) -> Episode:
        ep = Episode(id=str(uuid.uuid4()), task=task, kind=kind, success=success, used_tools=used_tools or [], used_skills=used_skills or [], execution_time=execution_time, error=error, summary=summary, created_at=datetime.now(UTC).isoformat())
        with self._connect() as con:
            con.execute("INSERT INTO episodes(id, task, kind, success, used_tools, used_skills, execution_time, error, summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (ep.id, ep.task, ep.kind, int(ep.success), json.dumps(ep.used_tools), json.dumps(ep.used_skills), ep.execution_time, ep.error, ep.summary, ep.created_at))
        return ep
    def list_recent(self, limit: int = 20) -> list[Episode]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_episode(r) for r in rows]
    def successful_tool_sequences(self, min_length: int = 2, limit: int = 200) -> list[list[str]]:
        return [ep.used_tools for ep in self.list_recent(limit) if ep.success and len(ep.used_tools) >= min_length]
    def _row_to_episode(self, row) -> Episode:
        return Episode(id=row["id"], task=row["task"], kind=row["kind"], success=bool(row["success"]), used_tools=json.loads(row["used_tools"]), used_skills=json.loads(row["used_skills"]), execution_time=float(row["execution_time"]), error=row["error"], summary=row["summary"], created_at=row["created_at"])
