from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ROOT_DIR, PROTECTED_CORE_FILES


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    action: str
    reason: str
    required_approval: bool = False
    risks: list[str] | None = None

    def model_dump(self) -> dict:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "required_approval": self.required_approval,
            "risks": self.risks or [],
        }


class SafetyGate:
    """Central permission gate for core, tool, skill and external actions."""

    CRITICAL_ACTIONS = {
        "shell",
        "network",
        "package_install",
        "process_start",
        "external_api",
        "core_modify",
        "secret_access",
    }

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir.resolve()

    def evaluate(self, action: str, paths: Iterable[str | Path] | None = None, approved: bool = False) -> SafetyDecision:
        risks: list[str] = []
        normalized_action = action.strip().lower()

        if normalized_action in self.CRITICAL_ACTIONS and not approved:
            return SafetyDecision(False, action, "critical action requires explicit user approval", True, [normalized_action])

        for raw_path in paths or []:
            path = Path(raw_path)
            resolved = (self.root_dir / path).resolve() if not path.is_absolute() else path.resolve()
            try:
                resolved.relative_to(self.root_dir)
            except ValueError:
                risks.append(f"path outside project: {resolved}")
            if resolved.parent.name == "core" and resolved.name in PROTECTED_CORE_FILES and not approved:
                risks.append(f"protected core file requires approval: core/{resolved.name}")

        if risks:
            return SafetyDecision(False, action, "safety risks detected", True, risks)
        return SafetyDecision(True, action, "allowed", False, [])
