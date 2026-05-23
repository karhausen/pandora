from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.short_term_path = memory_dir / "short_term.json"
        self.long_term_path = memory_dir / "long_term.sqlite"
        self.episodic_path = memory_dir / "episodic.sqlite"
        self.semantic_path = memory_dir / "semantic.sqlite"

    def initialize(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.short_term_path.exists():
            self.short_term_path.write_text("{}", encoding="utf-8")
        self._init_db(self.long_term_path, "long_term")
        self._init_db(self.episodic_path, "episodes")
        self._init_db(self.semantic_path, "facts")

    def _init_db(self, path: Path, table: str) -> None:
        with sqlite3.connect(path) as con:
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, value TEXT, created_at REAL)"
            )
            con.commit()

    def set_short_term(self, key: str, value: Any) -> None:
        data = self.get_short_term_all()
        data[key] = value
        self.short_term_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_short_term_all(self) -> dict[str, Any]:
        if not self.short_term_path.exists():
            return {}
        return json.loads(self.short_term_path.read_text(encoding="utf-8") or "{}")

    def add_episode(self, key: str, value: Any) -> None:
        self._insert(self.episodic_path, "episodes", key, value)

    def add_long_term(self, key: str, value: Any) -> None:
        self._insert(self.long_term_path, "long_term", key, value)

    def _insert(self, db_path: Path, table: str, key: str, value: Any) -> None:
        with sqlite3.connect(db_path) as con:
            con.execute(
                f"INSERT INTO {table}(key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )
            con.commit()

    def healthcheck(self) -> bool:
        self.initialize()
        _ = self.get_short_term_all()
        return self.short_term_path.exists() and self.long_term_path.exists()
