from __future__ import annotations

import json
from pathlib import Path

from .config import EXECUTION_POLICY_FILE, LEGACY_EXECUTION_POLICY_FILE
from .models import ExecutionPolicy, ExecutionPolicyName


class ExecutionPolicyManager:
    def __init__(self, path: Path = EXECUTION_POLICY_FILE):
        self.path = path

    def load(self) -> dict:
        path = self.path
        if not path.exists() and path == EXECUTION_POLICY_FILE and LEGACY_EXECUTION_POLICY_FILE.exists():
            path = LEGACY_EXECUTION_POLICY_FILE
        return json.loads(path.read_text(encoding="utf-8"))

    def get_for_tool(self, tool_id: str) -> ExecutionPolicy:
        data = self.load()
        policy_name = data.get("tool_overrides", {}).get(tool_id, data.get("default_policy", "restricted"))
        raw = data.get("policies", {}).get(policy_name, data["policies"]["restricted"])
        return ExecutionPolicy(name=ExecutionPolicyName(policy_name), **raw)

    def list(self) -> dict:
        return self.load()
