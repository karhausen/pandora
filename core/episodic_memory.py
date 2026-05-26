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
    def _connect(self): return sqlite3.connect(self.path)
    def _init_db(self):
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, task TEXT NOT NULL, kind TEXT NOT NULL, success INTEGER NOT NULL, used_tools TEXT NOT NULL, used_skills TEXT NOT NULL, execution_time REAL NOT NULL, error TEXT, summary TEXT, created_at TEXT NOT NULL)''')
    def record(self, task, kind, success, used_tools=None, used_skills=None, execution_time=0.0, error=None, summary=None):
        ep = Episode(id=str(uuid.uuid4()), task=task, kind=kind, success=success, used_tools=used_tools or [], used_skills=used_skills or [], execution_time=execution_time, error=error, summary=summary, created_at=datetime.now(UTC).isoformat())
        with self._connect() as con:
            con.execute("INSERT INTO episodes(id, task, kind, success, used_tools, used_skills, execution_time, error, summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (ep.id, ep.task, ep.kind, int(ep.success), json.dumps(ep.used_tools), json.dumps(ep.used_skills), ep.execution_time, ep.error, ep.summary, ep.created_at))
        return ep
    def list_recent(self, limit=20):
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [Episode(id=r["id"], task=r["task"], kind=r["kind"], success=bool(r["success"]), used_tools=json.loads(r["used_tools"]), used_skills=json.loads(r["used_skills"]), execution_time=float(r["execution_time"]), error=r["error"], summary=r["summary"], created_at=r["created_at"]) for r in rows]
