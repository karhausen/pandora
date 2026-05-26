from __future__ import annotations
import json
from datetime import datetime, UTC
from .config import REFLECTION_LOG

class ReflectionLogger:
    def __init__(self, path=REFLECTION_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def record(self, event: dict):
        event = dict(event); event["created_at"] = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    def tail(self, limit=20):
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()[-limit:] if line.strip()]

class ReflectionEngine:
    def __init__(self):
        self.logger = ReflectionLogger()
    def reflect_tool_result(self, tool_id, success, execution_time, error=None):
        self.logger.record({"type":"tool_result","tool_id":tool_id,"success":success,"execution_time":execution_time,"error":error})
    def reflect_skill_result(self, skill_id, success, execution_time, error=None):
        self.logger.record({"type":"skill_result","skill_id":skill_id,"success":success,"execution_time":execution_time,"error":error})
