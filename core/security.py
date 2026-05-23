from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROTECTED_CORE_FILES = {
    "heartbeat.py",
    "rollback.py",
    "recovery.py",
    "security.py",
    "config.py",
}


@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    reason: str


class SecurityPolicy:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def path_allowed(self, path: Path, allowed_root: Path) -> SecurityDecision:
        target = path.resolve()
        root = allowed_root.resolve()
        if root == target or root in target.parents:
            return SecurityDecision(True, "path allowed")
        return SecurityDecision(False, f"path outside allowed root: {target}")

    def core_patch_allowed(self, file_name: str, user_approved: bool = False) -> SecurityDecision:
        if file_name in PROTECTED_CORE_FILES and not user_approved:
            return SecurityDecision(False, f"protected core file requires explicit approval: {file_name}")
        return SecurityDecision(True, "core patch allowed")
