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
        event = dict(event)
        event["created_at"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]


class ReflectionEngine:
    def __init__(self):
        self.logger = ReflectionLogger()

    def reflect_tool_result(self, tool_id: str, success: bool, execution_time: float, error: str | None = None) -> ReflectionInsight:
        if not success:
            insight = ReflectionInsight(
                kind="tool_failure",
                severity="warning",
                message=f"Tool '{tool_id}' failed.",
                suggested_action="Inspect error history and consider disabling or improving the tool.",
                data={"tool_id": tool_id, "error": error},
            )
        elif execution_time > 2.0:
            insight = ReflectionInsight(
                kind="tool_slow",
                severity="info",
                message=f"Tool '{tool_id}' was slow.",
                suggested_action="Consider optimization if this repeats.",
                data={"tool_id": tool_id, "execution_time": execution_time},
            )
        else:
            insight = ReflectionInsight(
                kind="tool_success",
                severity="info",
                message=f"Tool '{tool_id}' completed successfully.",
                data={"tool_id": tool_id, "execution_time": execution_time},
            )

        self.logger.record({"type": "reflection_insight", **insight.model_dump(mode="json")})
        return insight

    def reflect_skill_result(self, skill_id: str, success: bool, execution_time: float, error: str | None = None) -> ReflectionInsight:
        if not success:
            insight = ReflectionInsight(
                kind="skill_failure",
                severity="warning",
                message=f"Skill '{skill_id}' failed.",
                suggested_action="Inspect failing step and dependencies.",
                data={"skill_id": skill_id, "error": error},
            )
        else:
            insight = ReflectionInsight(
                kind="skill_success",
                severity="info",
                message=f"Skill '{skill_id}' completed successfully.",
                data={"skill_id": skill_id, "execution_time": execution_time},
            )

        self.logger.record({"type": "reflection_insight", **insight.model_dump(mode="json")})
        return insight
