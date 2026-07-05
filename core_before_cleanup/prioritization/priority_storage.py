from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .priority_schema import ImprovementCandidate, PriorityScore, utc_now

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "prioritization" / "priority_queue.db"


class PriorityStorage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_pattern_id TEXT,
                    candidate_type TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    recommendation_hint TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS scores (
                    candidate_id TEXT PRIMARY KEY,
                    scored_at TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    level TEXT NOT NULL,
                    factors_json TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    explanation TEXT NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS priority_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
            """)

    def save(self, rows: list[tuple[ImprovementCandidate, PriorityScore]]) -> dict[str, Any]:
        with self._connect() as con:
            for candidate, score in rows:
                cd = candidate.as_dict(); sd = score.as_dict()
                con.execute("""INSERT OR REPLACE INTO candidates
                    (candidate_id,created_at,title,description,source_pattern_id,candidate_type,evidence_json,recommendation_hint)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (cd["candidate_id"], cd["created_at"], cd["title"], cd["description"], cd["source_pattern_id"], cd["candidate_type"], json.dumps(cd["evidence"], ensure_ascii=False), cd["recommendation_hint"]),
                )
                con.execute("""INSERT OR REPLACE INTO scores
                    (candidate_id,scored_at,total_score,level,factors_json,weights_json,explanation)
                    VALUES (?,?,?,?,?,?,?)""",
                    (sd["candidate_id"], sd["scored_at"], sd["total_score"], sd["level"], json.dumps(sd["factors"], ensure_ascii=False), json.dumps(sd["weights"], ensure_ascii=False), sd["explanation"]),
                )
            con.execute("INSERT OR REPLACE INTO priority_runs (run_id,created_at,summary_json) VALUES (?,?,?)", (f"run_{utc_now()}", utc_now(), json.dumps({"saved": len(rows)}, ensure_ascii=False)))
        return {"kind": "prioritization_save", "version": "28.8", "saved": len(rows), "storage": str(self.db_path)}

    def queue(self, limit: int = 50, level: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        sql = """SELECT c.candidate_id,c.created_at,c.title,c.description,c.source_pattern_id,c.candidate_type,c.evidence_json,c.recommendation_hint,
                 s.scored_at,s.total_score,s.level,s.factors_json,s.weights_json,s.explanation
                 FROM candidates c JOIN scores s ON c.candidate_id=s.candidate_id"""
        params: list[Any] = []
        if level:
            sql += " WHERE s.level = ?"; params.append(level)
        sql += " ORDER BY s.total_score DESC, c.created_at DESC LIMIT ?"; params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        out=[]
        for r in rows:
            out.append({
                "candidate_id": r[0], "created_at": r[1], "title": r[2], "description": r[3], "source_pattern_id": r[4], "candidate_type": r[5],
                "evidence": json.loads(r[6] or "{}"), "recommendation_hint": r[7],
                "score": {"scored_at": r[8], "total_score": r[9], "level": r[10], "factors": json.loads(r[11] or "{}"), "weights": json.loads(r[12] or "{}"), "explanation": r[13]},
                "creates_proposals": False,
            })
        return out

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT run_id,created_at,summary_json FROM priority_runs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
        return [{"run_id": r[0], "created_at": r[1], "summary": json.loads(r[2] or "{}")} for r in rows]
