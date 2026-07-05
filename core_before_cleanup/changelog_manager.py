from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

from .config import CHANGELOG_FILE


class ChangelogManager:
    def __init__(self, path: Path = CHANGELOG_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def ensure(self) -> str:
        if not self.path.exists():
            self.path.write_text("# Changelog\n\n", encoding="utf-8")
        return self.path.read_text(encoding="utf-8")

    def add_entry(self, version: str, title: str, items: list[str]) -> str:
        self.ensure()
        date = datetime.now(UTC).date().isoformat()
        body = f"\n## {version} - {date} - {title}\n\n"
        for item in items:
            body += f"- {item}\n"
        current = self.path.read_text(encoding="utf-8")
        if f"## {version} " not in current:
            self.path.write_text(current.rstrip() + "\n" + body, encoding="utf-8")
        return self.path.read_text(encoding="utf-8")

    def read(self) -> str:
        return self.ensure()
