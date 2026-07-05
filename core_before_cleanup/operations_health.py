from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable


class OperationsHealthService:
    """Lightweight health and diagnostics view for Pandora operations.

    This service is read-only. It aggregates status checks that are useful for
    day-to-day operation and release validation without changing tools, skills,
    knowledge or core files.
    """

    version = "mvp-24.11-operations-health-system-diagnostics"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.parent

    def status(self) -> dict[str, Any]:
        checks = self.run_checks()
        counts = self._counts(checks)
        overall = self._overall(counts)
        return {
            "kind": "operations_health_status",
            "version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "overall": overall,
            "counts": counts,
            "checks": checks,
            "recommendations": self._recommendations(checks),
            "safety": self.safety(),
        }

    def run_checks(self) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        checks.extend(self._filesystem_checks())
        checks.extend(self._service_checks())
        checks.extend(self._registration_checks())
        checks.extend(self._web_checks())
        return checks

    def _filesystem_checks(self) -> list[dict[str, Any]]:
        required_files = ["main.py", "core/api.py", "web/shared.css", "README.md"]
        required_dirs = ["core", "web", "docs", "tests"]
        checks: list[dict[str, Any]] = []
        for rel in required_files:
            checks.append(self._check(
                id=f"file:{rel}",
                area="filesystem",
                title=f"Required file {rel}",
                ok=(self.root / rel).is_file(),
                detail=str(self.root / rel),
                severity="error",
            ))
        for rel in required_dirs:
            checks.append(self._check(
                id=f"dir:{rel}",
                area="filesystem",
                title=f"Required directory {rel}",
                ok=(self.root / rel).is_dir(),
                detail=str(self.root / rel),
                severity="error",
            ))
        return checks

    def _service_checks(self) -> list[dict[str, Any]]:
        service_calls: list[tuple[str, str, Callable[[], Any]]] = []
        try:
            from .core_status import CoreStatusService
            service_calls.append(("core_status", "Core status service", lambda: CoreStatusService().status()))
        except Exception as exc:
            return [self._check("service:core_status:import", "services", "Core status import", False, str(exc), "error")]

        optional_services = [
            ("action_inbox", "Unified Action Inbox", ".unified_action_inbox", "UnifiedActionInboxService", "dashboard"),
            ("workflows", "Workflow Dashboard", ".workflow_dashboard", "WorkflowDashboardService", "dashboard"),
            ("night_review", "Night Review", ".night_review_engine", "NightReviewEngine", "status"),
            ("review_scheduler", "Review Scheduler", ".review_scheduler", "ReviewSchedulerService", "status"),
            ("release", "Release Manager", ".release_manager", "ReleaseManager", "status"),
        ]
        for sid, title, module_name, class_name, method_name in optional_services:
            try:
                module = __import__(f"core{module_name}", fromlist=[class_name])
                cls = getattr(module, class_name)
                service_calls.append((sid, title, lambda cls=cls, method_name=method_name: getattr(cls(), method_name)()))
            except Exception as exc:
                service_calls.append((sid, title, lambda exc=exc: (_ for _ in ()).throw(exc)))

        checks: list[dict[str, Any]] = []
        for sid, title, fn in service_calls:
            try:
                result = fn()
                ok = isinstance(result, dict) or isinstance(result, list)
                checks.append(self._check(f"service:{sid}", "services", title, ok, "responded" if ok else "unexpected result", "error"))
            except Exception as exc:
                checks.append(self._check(f"service:{sid}", "services", title, False, str(exc), "error"))
        return checks

    def _registration_checks(self) -> list[dict[str, Any]]:
        try:
            from .registration_validator import RegistrationValidator
            report = RegistrationValidator().validate()
            ok = bool(report.get("ok", False))
            return [self._check(
                id="registration:strict",
                area="registration",
                title="Registration validation",
                ok=ok,
                detail="OK" if ok else str(report.get("summary") or report.get("errors") or report),
                severity="error",
                payload={"summary": report.get("summary"), "ok": ok},
            )]
        except Exception as exc:
            return [self._check("registration:strict", "registration", "Registration validation", False, str(exc), "error")]

    def _web_checks(self) -> list[dict[str, Any]]:
        expected = [
            "/", "/operations-cockpit", "/action-inbox", "/workflow-dashboard",
            "/night-review", "/review-scheduler", "/operations-health",
        ]
        checks: list[dict[str, Any]] = []
        for route in expected:
            if route == "/":
                rel = "web/index.html"
            else:
                rel = "web/" + route.strip("/") + ".html"
            checks.append(self._check(
                id=f"web:{route}",
                area="web",
                title=f"Web page {route}",
                ok=(self.root / rel).is_file(),
                detail=rel,
                severity="warning" if route == "/operations-health" else "error",
            ))
        return checks

    def safety(self) -> dict[str, bool]:
        return {
            "read_only": True,
            "runs_user_actions": False,
            "changes_core": False,
            "writes_release_artifacts": False,
            "safe_for_dashboard": True,
        }

    def _check(self, id: str, area: str, title: str, ok: bool, detail: str, severity: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "id": id,
            "area": area,
            "title": title,
            "status": "ok" if ok else ("error" if severity == "error" else "warning"),
            "ok": bool(ok),
            "severity": severity,
            "detail": detail,
            "payload": payload or {},
        }

    def _counts(self, checks: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "total": len(checks),
            "ok": sum(1 for c in checks if c.get("status") == "ok"),
            "warning": sum(1 for c in checks if c.get("status") == "warning"),
            "error": sum(1 for c in checks if c.get("status") == "error"),
        }

    def _overall(self, counts: dict[str, int]) -> str:
        if counts.get("error", 0):
            return "error"
        if counts.get("warning", 0):
            return "warning"
        return "ok"

    def _recommendations(self, checks: list[dict[str, Any]]) -> list[dict[str, str]]:
        recommendations: list[dict[str, str]] = []
        for check in checks:
            if check.get("status") == "ok":
                continue
            recommendations.append({
                "level": check.get("status", "warning"),
                "area": check.get("area", "unknown"),
                "title": check.get("title", "Unknown check"),
                "next_step": self._next_step(check),
            })
        return recommendations

    def _next_step(self, check: dict[str, Any]) -> str:
        area = check.get("area")
        if area == "filesystem":
            return "Fehlende Projektdatei oder Ordner aus dem letzten vollständigen Release wiederherstellen."
        if area == "registration":
            return "registration-validate --strict ausführen und fehlende CLI/API/GUI-Handler reparieren."
        if area == "web":
            return "HTML/JS/CSS-Route ergänzen oder die Webroute aus der Navigation entfernen."
        if area == "services":
            return "Service-Import und Statusmethode prüfen; Fehler im Detailfeld ansehen."
        return "Check-Details prüfen."
