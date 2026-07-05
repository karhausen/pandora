from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .operations_health import OperationsHealthService
from .config import ROOT_DIR


class OperationsIssueDetector:
    """Detect operational issues from health checks and configuration signals.

    Read-only detector: it does not fix or execute anything.
    """

    version = "mvp-24.12-operations-issue-actions"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT_DIR
        self.health = OperationsHealthService(self.root)

    def scan(self) -> dict[str, Any]:
        checks = self.health.run_checks()
        issues = [self._from_check(c) for c in checks if c.get("status") != "ok"]
        issues.extend(self._configuration_issues())
        issues = self._dedupe(issues)
        counts = {
            "total": len(issues),
            "critical": sum(1 for i in issues if i.get("priority") == "critical"),
            "high": sum(1 for i in issues if i.get("priority") == "high"),
            "medium": sum(1 for i in issues if i.get("priority") == "medium"),
            "low": sum(1 for i in issues if i.get("priority") == "low"),
        }
        return {
            "kind": "operations_issue_scan",
            "version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "issues": issues,
            "safety": {"read_only": True, "auto_fix": False, "creates_actions": False},
        }

    def status(self) -> dict[str, Any]:
        report = self.scan()
        return {
            "kind": "operations_issue_status",
            "version": self.version,
            "generated_at": report["generated_at"],
            "counts": report["counts"],
            "open_issues": report["issues"][:25],
            "safety": report["safety"],
        }

    def show(self, issue_id: str) -> dict[str, Any]:
        for issue in self.scan()["issues"]:
            if issue.get("id") == issue_id:
                return {"kind": "operations_issue_detail", "found": True, "issue": issue}
        return {"kind": "operations_issue_detail", "found": False, "issue_id": issue_id}

    def _from_check(self, check: dict[str, Any]) -> dict[str, Any]:
        severity = check.get("severity") or check.get("status") or "warning"
        priority = "high" if severity == "error" else "medium"
        if check.get("area") == "registration":
            priority = "critical" if check.get("status") == "error" else "high"
        return {
            "id": f"ops:{check.get('id', 'unknown')}",
            "title": check.get("title") or "Operations issue",
            "area": check.get("area") or "operations",
            "type": "health_check_issue",
            "priority": priority,
            "status": "open",
            "detail": check.get("detail") or "",
            "recommended_action": self._recommended_action(check),
            "source": "operations_health",
            "source_check": check,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _configuration_issues(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        env_example = self.root / ".env.example"
        if env_example.exists():
            text = env_example.read_text(encoding="utf-8", errors="ignore")
            if "OBSIDIAN_VAULT_PATH" in text and "OBSIDIAN_VAULT_ENABLED" not in text:
                issues.append(self._simple("ops:config:obsidian-env", "Obsidian .env Beispiel unvollständig", "configuration", "medium", "OBSIDIAN_VAULT_ENABLED fehlt in .env.example", "Obsidian-Konfiguration prüfen und Beispiel ergänzen."))
        if not (self.root / "release.json").exists():
            issues.append(self._simple("ops:release:missing-release-json", "release.json fehlt", "release", "medium", "Release-Metadaten fehlen.", "Release-Metadaten erzeugen."))
        return issues

    def _simple(self, issue_id: str, title: str, area: str, priority: str, detail: str, action: str) -> dict[str, Any]:
        return {"id": issue_id, "title": title, "area": area, "type": "configuration_issue", "priority": priority, "status": "open", "detail": detail, "recommended_action": action, "source": "operations_issue_detector", "created_at": datetime.now(UTC).isoformat()}

    def _recommended_action(self, check: dict[str, Any]) -> str:
        area = check.get("area")
        if area == "web":
            return "Webroute/HTML/JS/CSS prüfen und fehlende Route ergänzen oder Navigation korrigieren."
        if area == "registration":
            return "registration-validate --strict ausführen und fehlende Handler/Routen reparieren."
        if area == "services":
            return "Service-Import und Statusmethode prüfen; Detailfehler analysieren."
        if area == "filesystem":
            return "Fehlende Projektdatei oder Ordner aus dem letzten vollständigen Release wiederherstellen."
        return "Issue prüfen und nächsten sicheren Schritt planen."

    def _dedupe(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for issue in issues:
            key = str(issue.get("id"))
            if key in seen:
                continue
            seen.add(key)
            result.append(issue)
        return result
