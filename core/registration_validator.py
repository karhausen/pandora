from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ValidationIssue:
    area: str
    severity: str
    message: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


class RegistrationValidator:
    """Validate that CLI, API and Web-GUI registrations are internally consistent.

    This is a release guard. It does not execute user workflows or call live LLMs.
    It only checks that registered handlers/routes can be resolved and that GUI
    fetch calls point to known API route prefixes.
    """

    def __init__(self, root_dir: Path = ROOT):
        self.root_dir = root_dir
        self.main_path = root_dir / "main.py"
        self.web_dir = root_dir / "web"

    def validate(self) -> dict[str, Any]:
        issues: list[ValidationIssue] = []
        cli = self.validate_cli()
        api = self.validate_api()
        gui = self.validate_gui(api_routes=api.get("routes", []))
        issues.extend(ValidationIssue(**item) for item in cli.get("issues", []))
        issues.extend(ValidationIssue(**item) for item in api.get("issues", []))
        issues.extend(ValidationIssue(**item) for item in gui.get("issues", []))
        return {
            "kind": "registration_validation_report",
            "version": "mvp-23.3.3-registration-validation",
            "ok": not any(item.severity == "error" for item in issues),
            "issue_count": len(issues),
            "error_count": sum(1 for item in issues if item.severity == "error"),
            "warning_count": sum(1 for item in issues if item.severity == "warning"),
            "checks": {
                "cli": cli,
                "api": api,
                "gui": gui,
            },
            "issues": [item.as_dict() for item in issues],
        }

    def validate_cli(self) -> dict[str, Any]:
        src = self.main_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("cmd_")
        }
        referenced = set(re.findall(r"set_defaults\(func=(cmd_[A-Za-z0-9_]+)", src))
        commands = re.findall(r"sub\.add_parser\(\s*[\"']([^\"']+)[\"']", src)
        issues: list[ValidationIssue] = []
        missing = sorted(referenced - defined)
        for name in missing:
            issues.append(ValidationIssue("cli", "error", "CLI parser references a missing handler", {"handler": name}))
        orphaned = sorted(defined - referenced)
        for name in orphaned:
            # Orphaned command functions are not fatal; they may be helper commands in development.
            issues.append(ValidationIssue("cli", "warning", "Command handler is defined but not registered", {"handler": name}))
        return {
            "ok": not missing,
            "command_count": len(commands),
            "handler_reference_count": len(referenced),
            "handler_definition_count": len(defined),
            "missing_handlers": missing,
            "orphaned_handlers": orphaned,
            "issues": [issue.as_dict() for issue in issues],
        }

    def validate_api(self) -> dict[str, Any]:
        issues: list[ValidationIssue] = []
        try:
            module = importlib.import_module("core.api")
            app = getattr(module, "app")
            routes = []
            for route in app.routes:
                path = getattr(route, "path", None)
                methods = sorted(getattr(route, "methods", []) or [])
                endpoint = getattr(route, "endpoint", None)
                endpoint_name = getattr(endpoint, "__name__", None)
                if path:
                    routes.append({"path": path, "methods": methods, "endpoint": endpoint_name})
                    if endpoint is None:
                        issues.append(ValidationIssue("api", "error", "API route has no endpoint", {"path": path}))
            return {
                "ok": not any(item.severity == "error" for item in issues),
                "route_count": len(routes),
                "routes": routes,
                "issues": [issue.as_dict() for issue in issues],
            }
        except Exception as exc:  # pragma: no cover - defensive release guard
            issues.append(ValidationIssue("api", "error", "API module could not be imported", {"error": f"{type(exc).__name__}: {exc}"}))
            return {"ok": False, "route_count": 0, "routes": [], "issues": [issue.as_dict() for issue in issues]}

    def validate_gui(self, *, api_routes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        route_paths = [str(route.get("path")) for route in api_routes or []]
        normalized_routes = {self._normalize_api_path(path) for path in route_paths if path.startswith("/api/")}
        fetches = self._extract_fetches()
        issues: list[ValidationIssue] = []
        unknown: list[dict[str, str]] = []
        for item in fetches:
            normalized = self._normalize_fetch_path(item["path"])
            if not normalized:
                continue
            if not self._fetch_matches_route(normalized, normalized_routes):
                detail = {"file": item["file"], "fetch": item["path"], "normalized": normalized}
                unknown.append(detail)
                issues.append(ValidationIssue("gui", "warning", "GUI fetch path has no obvious matching API route", detail))
        return {
            "ok": not any(item.severity == "error" for item in issues),
            "fetch_count": len(fetches),
            "unknown_fetch_count": len(unknown),
            "unknown_fetches": unknown,
            "issues": [issue.as_dict() for issue in issues],
        }

    def _extract_fetches(self) -> list[dict[str, str]]:
        if not self.web_dir.exists():
            return []
        pattern = re.compile(r"fetch\(\s*([`\"'])(/api/.*?)(?:\1|\$\{)")
        fetches: list[dict[str, str]] = []
        for path in sorted(self.web_dir.rglob("*.js")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in pattern.finditer(text):
                raw = match.group(2).split("${", 1)[0]
                raw = raw.split("?", 1)[0]
                fetches.append({"file": path.relative_to(self.root_dir).as_posix(), "path": raw.rstrip("/") or "/"})
        return fetches

    def _normalize_api_path(self, path: str) -> str:
        return re.sub(r"\{[^}]+\}", "*", path.rstrip("/"))

    def _normalize_fetch_path(self, path: str) -> str:
        if not path.startswith("/api/"):
            return ""
        # Dynamic template strings are captured only up to the static prefix. Keep that prefix.
        return path.rstrip("/")

    def _fetch_matches_route(self, fetch_path: str, route_paths: set[str]) -> bool:
        if fetch_path in route_paths:
            return True
        for route in route_paths:
            if "*" in route:
                prefix = route.split("*", 1)[0].rstrip("/")
                if fetch_path == prefix or fetch_path.startswith(prefix + "/") or prefix.startswith(fetch_path + "/"):
                    return True
            if route.startswith(fetch_path + "/"):
                # Template fetch captured only a stable prefix.
                return True
        return False
