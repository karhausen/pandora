from __future__ import annotations


class LLMGovernance:
    protected_actions = {
        "modify_core",
        "disable_heartbeat",
        "disable_rollback",
        "escalate_permissions",
        "execute_shell",
    }

    def validate_action(self, action: str) -> dict:
        if action in self.protected_actions:
            return {
                "allowed": False,
                "reason": f"Action '{action}' requires explicit user approval and cannot be executed by LLM directly.",
            }
        return {"allowed": True, "reason": "allowed"}
