from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .observation_schema import ObservationEvent, utc_now

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "observations" / "observations.db"


class ObservationStorage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS observation_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    duration_ms INTEGER,
                    metadata_json TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS observation_statistics (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def add_event(self, event: ObservationEvent) -> dict[str, Any]:
        data = event.as_dict()
        with self._connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO observation_events
                (event_id,timestamp,component,event_type,success,severity,message,duration_ms,metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (data["event_id"], data["timestamp"], data["component"], data["event_type"], int(data["success"]), data["severity"], data["message"], data["duration_ms"], json.dumps(data["metadata"], ensure_ascii=False)),
            )
        return data

    def list_events(self, limit: int = 50, component: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 500))
        sql = "SELECT event_id,timestamp,component,event_type,success,severity,message,duration_ms,metadata_json FROM observation_events"
        params: list[Any] = []
        if component:
            sql += " WHERE component = ?"
            params.append(component)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def _row_to_event(self, r) -> dict[str, Any]:
        return {
            "event_id": r[0], "timestamp": r[1], "component": r[2], "event_type": r[3],
            "success": bool(r[4]), "severity": r[5], "message": r[6] or "", "duration_ms": r[7],
            "metadata": json.loads(r[8] or "{}"),
        }

    def statistics(self) -> dict[str, Any]:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM observation_events").fetchone()[0]
            failures = con.execute("SELECT COUNT(*) FROM observation_events WHERE success=0").fetchone()[0]
            by_component = con.execute("SELECT component, COUNT(*) FROM observation_events GROUP BY component ORDER BY COUNT(*) DESC").fetchall()
            by_type = con.execute("SELECT event_type, COUNT(*) FROM observation_events GROUP BY event_type ORDER BY COUNT(*) DESC").fetchall()
            avg_duration = con.execute("SELECT AVG(duration_ms) FROM observation_events WHERE duration_ms IS NOT NULL").fetchone()[0]
        return {
            "kind": "observation_statistics",
            "version": "28.6",
            "total_events": total,
            "failed_events": failures,
            "success_rate": 1.0 if total == 0 else round((total - failures) / total, 4),
            "by_component": dict(by_component),
            "by_event_type": dict(by_type),
            "avg_duration_ms": round(avg_duration, 2) if avg_duration is not None else None,
        }

    def export(self, limit: int = 500) -> dict[str, Any]:
        return {"kind": "observation_export", "version": "28.6", "exported_at": utc_now(), "events": self.list_events(limit=limit), "statistics": self.statistics()}
