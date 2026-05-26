from __future__ import annotations

import sqlite3
from pathlib import Path
from .config import TOOL_RUNTIME_DB


class ToolRuntimeDB:
    def __init__(self, path: Path = TOOL_RUNTIME_DB):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS tool_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                execution_time REAL NOT NULL,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS tool_stats (
                tool_id TEXT PRIMARY KEY,
                runs INTEGER NOT NULL,
                successes INTEGER NOT NULL,
                failures INTEGER NOT NULL,
                avg_execution_time REAL NOT NULL
            )
            """)

    def record_run(self, tool_id: str, success: bool, execution_time: float, error: str | None) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO tool_runs(tool_id, success, execution_time, error) VALUES (?, ?, ?, ?)",
                (tool_id, int(success), execution_time, error),
            )
            rows = con.execute(
                "SELECT COUNT(*), SUM(success), AVG(execution_time) FROM tool_runs WHERE tool_id=?",
                (tool_id,),
            ).fetchone()
            runs = int(rows[0] or 0)
            successes = int(rows[1] or 0)
            failures = runs - successes
            avg_time = float(rows[2] or 0.0)
            con.execute(
                """
                INSERT INTO tool_stats(tool_id, runs, successes, failures, avg_execution_time)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    runs=excluded.runs,
                    successes=excluded.successes,
                    failures=excluded.failures,
                    avg_execution_time=excluded.avg_execution_time
                """,
                (tool_id, runs, successes, failures, avg_time),
            )

    def stats(self) -> list[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(row) for row in con.execute("SELECT * FROM tool_stats").fetchall()]
