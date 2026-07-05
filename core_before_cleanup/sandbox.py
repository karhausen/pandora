from __future__ import annotations

from .execution_policy import ExecutionPolicyManager
from .isolation_runner import IsolationRunner
from .permission_manager import PermissionManager
from .sandbox_log import SandboxLog
from .tool_registry import ToolRegistry


class Sandbox:
    def __init__(self):
        self.policies = ExecutionPolicyManager()
        self.permissions = PermissionManager()
        self.runner = IsolationRunner()
        self.log = SandboxLog()

    def run_tool(self, tool_id: str, payload: dict):
        registry = ToolRegistry()
        registry.discover()
        meta = registry.get(tool_id)
        if not meta:
            result = {
                "success": False,
                "tool_id": tool_id,
                "error": "Tool not found",
            }
            self.log.append(result)
            return result

        policy = self.policies.get_for_tool(tool_id)
        result = self.runner.run_tool_isolated(tool_id, meta.module, meta.function, payload, policy)
        data = result.model_dump(mode="json")
        self.log.append(data)
        return data

    def policy_report(self) -> dict:
        return self.policies.list()

    def logs(self, limit: int = 20) -> list[dict]:
        return self.log.list(limit)
