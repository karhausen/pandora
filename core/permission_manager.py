from __future__ import annotations

from pathlib import Path
from .config import ROOT_DIR
from .models import ExecutionPolicy


class PermissionManager:
    BLOCKED_TOKENS = [
        "subprocess",
        "os.system",
        "socket",
        "requests",
        "urllib.request",
        "httpx",
        "shutil.rmtree",
        "eval(",
        "exec(",
        "__import__(",
        "open(",
    ]

    def review_code(self, code: str, policy: ExecutionPolicy) -> dict:
        issues: list[str] = []
        warnings: list[str] = []

        for token in self.BLOCKED_TOKENS:
            if token in code:
                if token in {"subprocess", "os.system"} and policy.allow_shell:
                    warnings.append(f"Shell-related token allowed by policy: {token}")
                elif token in {"socket", "requests", "urllib.request", "httpx"} and policy.allow_network:
                    warnings.append(f"Network-related token allowed by policy: {token}")
                elif token == "open(" and policy.allow_write:
                    warnings.append("File open token allowed by write policy.")
                else:
                    issues.append(f"Blocked token for policy {policy.name.value}: {token}")

        return {
            "ok": not issues,
            "issues": issues,
            "warnings": warnings,
            "policy": policy.model_dump(mode="json"),
        }

    def path_allowed(self, path: Path, policy: ExecutionPolicy) -> bool:
        resolved = path.resolve()
        if not policy.allowed_paths:
            return False
        for rel in policy.allowed_paths:
            allowed = (ROOT_DIR / rel).resolve()
            if resolved == allowed or allowed in resolved.parents:
                return True
        return False
