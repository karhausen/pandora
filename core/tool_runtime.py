from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSummary:
    run_count: int
    success_count: int
    failure_count: int
    avg_runtime_ms: float
    last_error: str | None


class ToolRuntimeStore:
    """Persistente Telemetrie fuer Tool-Ausfuehrungen.

    MVP2-Ziel: Keine Magie, keine komplexe Observability. Nur harte Fakten:
    Laufzeit, Erfolg, Fehler, Payload-Groessen und Zeitstempel.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    runtime_ms INTEGER NOT NULL,
                    input_size INTEGER NOT NULL,
                    output_size INTEGER NOT NULL,
                    error TEXT,
                    exception_type TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    error TEXT NOT NULL,
                    traceback TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_stats (
                    tool_name TEXT PRIMARY KEY,
                    run_count INTEGER NOT NULL,
                    success_count INTEGER NOT NULL,
                    failure_count INTEGER NOT NULL,
                    avg_runtime_ms REAL NOT NULL,
                    last_error TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )

    def record_run(
        self,
        *,
        tool_name: str,
        success: bool,
        runtime_ms: int,
        payload: dict[str, Any],
        output: Any = None,
        error: str | None = None,
        exception_type: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        self.initialize()
        input_size = self._json_size(payload)
        output_size = self._json_size(output)
        now = time.time()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO tool_runs
                (tool_name, success, runtime_ms, input_size, output_size, error, exception_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tool_name, int(success), runtime_ms, input_size, output_size, error, exception_type, now),
            )
            if not success and error:
                con.execute(
                    """
                    INSERT INTO tool_failures (tool_name, error, traceback, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (tool_name, error, traceback_text, now),
                )
            current = self.summary(tool_name)
            if current.run_count == 0:
                run_count = 1
                success_count = 1 if success else 0
                failure_count = 0 if success else 1
                avg_runtime = float(runtime_ms)
            else:
                run_count = current.run_count + 1
                success_count = current.success_count + (1 if success else 0)
                failure_count = current.failure_count + (0 if success else 1)
                avg_runtime = ((current.avg_runtime_ms * current.run_count) + runtime_ms) / run_count
            con.execute(
                """
                INSERT INTO tool_stats
                (tool_name, run_count, success_count, failure_count, avg_runtime_ms, last_error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_name) DO UPDATE SET
                    run_count=excluded.run_count,
                    success_count=excluded.success_count,
                    failure_count=excluded.failure_count,
                    avg_runtime_ms=excluded.avg_runtime_ms,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (tool_name, run_count, success_count, failure_count, avg_runtime, error or current.last_error, now),
            )

    def summary(self, tool_name: str) -> RuntimeSummary:
        self.initialize()
        with self._connect() as con:
            row = con.execute(
                """
                SELECT run_count, success_count, failure_count, avg_runtime_ms, last_error
                FROM tool_stats WHERE tool_name=?
                """,
                (tool_name,),
            ).fetchone()
        if row is None:
            return RuntimeSummary(0, 0, 0, 0.0, None)
        return RuntimeSummary(int(row[0]), int(row[1]), int(row[2]), float(row[3]), row[4])

    def all_stats(self) -> list[dict[str, Any]]:
        self.initialize()
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT tool_name, run_count, success_count, failure_count, avg_runtime_ms, last_error, updated_at
                FROM tool_stats ORDER BY tool_name
                """
            ).fetchall()
        return [
            {
                "tool_name": r[0],
                "run_count": r[1],
                "success_count": r[2],
                "failure_count": r[3],
                "avg_runtime_ms": r[4],
                "last_error": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    def healthcheck(self) -> bool:
        self.initialize()
        with self._connect() as con:
            con.execute("SELECT 1").fetchone()
        return True

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _json_size(value: Any) -> int:
        try:
            return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception:
            return 0
