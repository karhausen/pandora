from __future__ import annotations

from pathlib import Path
from .config import PROMPTS_DIR


class PromptManager:
    def __init__(self, root: Path = PROMPTS_DIR):
        self.root = root

    def load(self, category: str, name: str) -> str:
        path = self.root / category / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path.read_text(encoding="utf-8")

    def list_prompts(self) -> list[dict]:
        if not self.root.exists():
            return []
        return [{"category": p.parent.name, "name": p.stem, "path": str(p)} for p in self.root.rglob("*.md")]
