from __future__ import annotations
import json
from pathlib import Path
from .config import SKILL_REGISTRY_FILE, SKILLS_DIR
from .models import SkillMeta

class SkillRegistry:
    def __init__(self, registry_file: Path = SKILL_REGISTRY_FILE):
        self.registry_file = registry_file
        self.skills: dict[str, SkillMeta] = {}
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.load()
    def load(self) -> None:
        if not self.registry_file.exists():
            self.skills = {}
            return
        self.skills = {k: SkillMeta.model_validate(v) for k, v in json.loads(self.registry_file.read_text(encoding="utf-8")).items()}
    def save(self) -> None:
        self.registry_file.write_text(json.dumps({k:v.model_dump(mode="json") for k,v in self.skills.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    def register(self, meta: SkillMeta) -> None:
        self.skills[meta.id] = meta
        self.save()
    def get(self, skill_id: str) -> SkillMeta | None:
        return self.skills.get(skill_id)
    def list(self) -> list[SkillMeta]:
        return list(self.skills.values())
    def discover(self) -> int:
        count = 0
        for path in SKILLS_DIR.glob("*.json"):
            try:
                meta = SkillMeta.model_validate(json.loads(path.read_text(encoding="utf-8")))
                self.register(meta)
                count += 1
            except Exception:
                continue
        return count
