from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .pattern_schema import RecognizedPattern, utc_now

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "patterns" / "patterns.db"


class PatternStorage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS recognized_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    trend TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    recommendation_hint TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS pattern_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
            """)

    def save_patterns(self, patterns: list[RecognizedPattern], source: str = "observation") -> dict[str, Any]:
        with self._connect() as con:
            for pattern in patterns:
                data = pattern.as_dict()
                con.execute(
                    """INSERT OR REPLACE INTO recognized_patterns
                    (pattern_id,created_at,pattern_type,title,description,confidence,trend,severity,evidence_json,recommendation_hint)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        data["pattern_id"], data["created_at"], data["pattern_type"], data["title"],
                        data["description"], data["confidence"], data["trend"], data["severity"],
                        json.dumps(data["evidence"], ensure_ascii=False), data["recommendation_hint"],
                    ),
                )
            con.execute(
                "INSERT OR REPLACE INTO pattern_runs (run_id,created_at,source,summary_json) VALUES (?,?,?,?)",
                (f"run_{utc_now()}", utc_now(), source, json.dumps({"patterns": len(patterns)}, ensure_ascii=False)),
            )
        return {"kind": "pattern_save", "version": "28.7", "saved": len(patterns), "storage": str(self.db_path)}

    def list_patterns(self, limit: int = 50, pattern_type: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        sql = "SELECT pattern_id,created_at,pattern_type,title,description,confidence,trend,severity,evidence_json,recommendation_hint FROM recognized_patterns"
        params: list[Any] = []
        if pattern_type:
            sql += " WHERE pattern_type = ?"
            params.append(pattern_type)
        sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    def _row_to_pattern(self, row) -> dict[str, Any]:
        return {
            "pattern_id": row[0],
            "created_at": row[1],
            "pattern_type": row[2],
            "title": row[3],
            "description": row[4],
            "confidence": row[5],
            "trend": row[6],
            "severity": row[7],
            "evidence": json.loads(row[8] or "{}"),
            "recommendation_hint": row[9] or "",
            "creates_proposals": False,
        }
