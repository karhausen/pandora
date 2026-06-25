from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".venv", "venv", "htmlcov", "tmp", "temp"}
RUNTIME_SUFFIXES = {".pyc", ".pyo", ".log"}
SECRET_NAMES = {".env", "llm_config.local.json"}
ALLOWED_SECRET_EXAMPLES = {".env.example", "llm_config.local.example.json"}


@dataclass(frozen=True)
class AuditFinding:
    level: str
    path: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "path": self.path, "message": self.message}


class ReleaseAudit:
    """Release safety audit for complete Pandora project exports."""

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()

    def run(self) -> dict[str, Any]:
        findings: list[AuditFinding] = []
        required = ["main.py", "core", "web", "docs", "tests", "requirements.txt"]
        for item in required:
            if not (self.root_dir / item).exists():
                findings.append(AuditFinding("error", item, "required release item is missing"))
        for path in self.root_dir.rglob("*"):
            rel = str(path.relative_to(self.root_dir)).replace("\\", "/")
            if any(part in RUNTIME_DIR_NAMES for part in path.parts):
                findings.append(AuditFinding("error", rel, "runtime/cache directory must not be part of release"))
                continue
            if path.is_file() and path.suffix in RUNTIME_SUFFIXES:
                findings.append(AuditFinding("error", rel, "runtime file must not be part of release"))
            if path.is_file() and path.name in SECRET_NAMES and path.name not in ALLOWED_SECRET_EXAMPLES:
                findings.append(AuditFinding("error", rel, "local secret/config file must not be part of release"))
        errors = [f for f in findings if f.level == "error"]
        return {
            "kind": "release_audit",
            "ok": not errors,
            "root": str(self.root_dir),
            "error_count": len(errors),
            "warning_count": len([f for f in findings if f.level == "warning"]),
            "findings": [f.as_dict() for f in findings[:200]],
        }
