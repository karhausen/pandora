from __future__ import annotations
import json
from datetime import datetime, UTC
from .config import REFLECTION_LOG
from .models import ReflectionInsight

class ReflectionLogger:
    def __init__(self, path=REFLECTION_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def record(self, event: dict) -> None:
        event = dict(event); event["created_at"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    def tail(self, limit: int = 20) -> list[dict]:
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:] if line.strip()]

class ReflectionEngine:
    def __init__(self):
        self.logger = ReflectionLogger()
    def reflect_tool_result(self, tool_id: str, success: bool, execution_time: float, error: str | None = None) -> ReflectionInsight:
        insight = ReflectionInsight(kind="tool_success" if success else "tool_failure", severity="info" if success else "warning", message=f"Tool '{tool_id}' {'completed successfully' if success else 'failed'}.", suggested_action=None if success else "Inspect error history.", data={"tool_id": tool_id, "execution_time": execution_time, "error": error})
        self.logger.record({"type": "reflection_insight", **insight.model_dump(mode="json")})
        return insight
    def reflect_skill_result(self, skill_id: str, success: bool, execution_time: float, error: str | None = None) -> ReflectionInsight:
        insight = ReflectionInsight(kind="skill_success" if success else "skill_failure", severity="info" if success else "warning", message=f"Skill '{skill_id}' {'completed successfully' if success else 'failed'}.", suggested_action=None if success else "Inspect failing step.", data={"skill_id": skill_id, "execution_time": execution_time, "error": error})
        self.logger.record({"type": "reflection_insight", **insight.model_dump(mode="json")})
        return insight
