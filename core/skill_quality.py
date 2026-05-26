from __future__ import annotations
import sqlite3
from pathlib import Path
from .config import SKILL_QUALITY_DB

class SkillQualityDB:
    def __init__(self, path: Path = SKILL_QUALITY_DB):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    def _connect(self): return sqlite3.connect(self.path)
    def _init_db(self):
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS skill_quality (skill_id TEXT PRIMARY KEY, runs INTEGER NOT NULL, successes INTEGER NOT NULL, failures INTEGER NOT NULL, avg_execution_time REAL NOT NULL, score REAL NOT NULL)''')
    def record(self, skill_id, success, execution_time):
        cur = self.get(skill_id)
        if cur:
            runs=cur["runs"]+1; successes=cur["successes"]+(1 if success else 0); failures=cur["failures"]+(0 if success else 1); avg=((cur["avg_execution_time"]*cur["runs"])+execution_time)/runs
        else:
            runs=1; successes=1 if success else 0; failures=0 if success else 1; avg=execution_time
        score=max(0.0,min(1.0,(successes/runs)-(failures/runs)-min(avg/10.0,0.25)))
        with self._connect() as con:
            con.execute('''INSERT INTO skill_quality(skill_id, runs, successes, failures, avg_execution_time, score) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(skill_id) DO UPDATE SET runs=excluded.runs, successes=excluded.successes, failures=excluded.failures, avg_execution_time=excluded.avg_execution_time, score=excluded.score''', (skill_id, runs, successes, failures, avg, score))
    def get(self, skill_id):
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM skill_quality WHERE skill_id=?", (skill_id,)).fetchone()
            return dict(row) if row else None
    def list(self):
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(row) for row in con.execute("SELECT * FROM skill_quality ORDER BY score DESC").fetchall()]
