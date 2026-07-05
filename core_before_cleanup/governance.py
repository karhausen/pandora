from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

from .config import GOVERNANCE_REPORT_FILE, PROTECTED_CORE_FILES, ROOT_DIR


class Governance:
    REQUIRED_PATHS = [
        "core",
        "tools",
        "skills",
        "memory",
        "tests",
        "docs",
        "README.md",
    ]

    REQUIRED_CORE_FILES = [
        "config.py",
        "heartbeat.py",
        "tool_registry.py",
        "tool_executor.py",
        "agent_loop.py",
    ]

    def check(self) -> dict:
        issues: list[str] = []
        warnings: list[str] = []

        for rel in self.REQUIRED_PATHS:
            if not (ROOT_DIR / rel).exists():
                issues.append(f"Missing required path: {rel}")

        for rel in self.REQUIRED_CORE_FILES:
            if not (ROOT_DIR / "core" / rel).exists():
                issues.append(f"Missing core file: core/{rel}")

        for protected in PROTECTED_CORE_FILES:
            path = ROOT_DIR / "core" / protected
            if not path.exists():
                warnings.append(f"Protected file not present in this MVP: core/{protected}")

        readme = ROOT_DIR / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            for heading in ["Quickstart", "CLI", "API", "Sicherheit", "Roadmap"]:
                if heading.lower() not in text.lower():
                    warnings.append(f"README may be missing section: {heading}")

        report = {
            "ok": not issues,
            "issues": issues,
            "warnings": warnings,
            "checked_at": datetime.now(UTC).isoformat(),
            "protected_files": sorted(PROTECTED_CORE_FILES),
        }

        GOVERNANCE_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOVERNANCE_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
