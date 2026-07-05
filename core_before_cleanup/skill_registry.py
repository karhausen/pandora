from __future__ import annotations
import json
from pathlib import Path
from .models import SkillMeta
from .config import SKILL_REGISTRY_FILE, SKILLS_DIR, LEGACY_SKILL_REGISTRY_FILE

class SkillRegistry:
    def __init__(self, registry_file: Path = SKILL_REGISTRY_FILE):
        self.registry_file = registry_file
        self.skills: dict[str, SkillMeta] = {}
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        path = self.registry_file
        if not path.exists() and path == SKILL_REGISTRY_FILE and LEGACY_SKILL_REGISTRY_FILE.exists():
            path = LEGACY_SKILL_REGISTRY_FILE
        if not path.exists():
            self.skills = {}
            return
        self.skills = {k: SkillMeta.model_validate(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}

    def save(self):
        self.registry_file.write_text(json.dumps({k: v.model_dump(mode="json") for k, v in self.skills.items()}, indent=2, ensure_ascii=False), encoding="utf-8")

    def register(self, meta: SkillMeta):
        self.skills[meta.id] = meta
        self.save()

    def get(self, skill_id: str):
        return self.skills.get(skill_id)

    def list(self):
        return list(self.skills.values())

    def discover(self):
        count = 0
        for path in SKILLS_DIR.glob("*.json"):
            try:
                meta = SkillMeta.model_validate(json.loads(path.read_text(encoding="utf-8")))
                self.register(meta)
                count += 1
            except Exception:
                continue
        return count
