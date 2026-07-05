from __future__ import annotations

import ast
import re
from typing import Any

from .models import ToolDesign


class ToolReviewAgent:
    """Policy-aware static review for generated tool proposals.

    This agent is intentionally local and deterministic. Cloud models may write
    code, but Pandora validates the result with local policy before a proposal can
    become VALIDATED.
    """

    ALWAYS_FORBIDDEN_IMPORT_ROOTS = {
        "subprocess", "socket", "ctypes", "multiprocessing", "shutil", "httpx", "requests"
    }
    SAFE_FORBIDDEN_IMPORT_ROOTS = {"urllib", "http"}
    NETWORK_ALLOWED_IMPORTS = {"urllib.request", "urllib.parse", "urllib.error", "json", "os"}
    FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open"}
    SECRET_PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
        re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*=\s*['\"][^'\"]{6,}['\"]"),
    ]

    def review(self, code: str, design: ToolDesign | dict[str, Any] | None = None) -> dict[str, Any]:
        design_data = self._design_dict(design)
        requires_network = bool(design_data.get("requires_network"))
        security_level = str(design_data.get("security_level", "SAFE")).upper()

        issues: list[str] = []
        warnings: list[str] = []
        policy = {
            "security_level": security_level,
            "requires_network": requires_network,
            "network_imports_allowed": bool(requires_network and security_level == "LIMITED"),
            "allowed_network_imports": sorted(self.NETWORK_ALLOWED_IMPORTS) if requires_network and security_level == "LIMITED" else [],
        }

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"ok": False, "risk": "HIGH", "issues": [f"SyntaxError: {exc}"], "warnings": warnings, "policy": policy}

        imports = self._imports(tree)
        for imported in imports:
            root = imported.split(".")[0]
            if root in self.ALWAYS_FORBIDDEN_IMPORT_ROOTS:
                issues.append(f"Forbidden import: {imported}")
                continue
            if root in self.SAFE_FORBIDDEN_IMPORT_ROOTS:
                if not (requires_network and security_level == "LIMITED" and imported in self.NETWORK_ALLOWED_IMPORTS):
                    issues.append(f"Forbidden import: {imported}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_CALLS:
                    issues.append(f"Forbidden call: {node.func.id}")
            if isinstance(node, ast.Call) and self._is_urlopen_call(node):
                if not (requires_network and security_level == "LIMITED"):
                    issues.append("urlopen is only allowed for LIMITED tools with requires_network=true")
                if not self._has_timeout_kw(node):
                    issues.append("Network call must set timeout keyword")

        if requires_network and security_level == "SAFE":
            issues.append("SAFE tool design must not require network")
        if requires_network and security_level == "LIMITED" and "urllib.request" not in imports:
            warnings.append("Network tool does not import urllib.request; verify implementation path.")

        for pattern in self.SECRET_PATTERNS:
            if pattern.search(code):
                issues.append("Potential hard-coded secret detected")
                break

        if "http://" in code:
            warnings.append("Plain HTTP URL detected; prefer HTTPS unless explicitly justified.")
        if requires_network and "os.environ" not in code and "os.getenv" not in code:
            warnings.append("Network tool should read external API keys/config from environment when needed.")

        issues = sorted(set(issues))
        warnings = sorted(set(warnings))
        risk = "HIGH" if issues else ("MEDIUM" if warnings or requires_network else "LOW")
        return {"ok": not issues, "risk": risk, "issues": issues, "warnings": warnings, "policy": policy}

    def _design_dict(self, design: ToolDesign | dict[str, Any] | None) -> dict[str, Any]:
        if design is None:
            return {}
        if hasattr(design, "model_dump"):
            return design.model_dump(mode="json")
        return dict(design)

    def _imports(self, tree: ast.AST) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                result.add(module)
        return result

    def _is_urlopen_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "urlopen":
            value = func.value
            if isinstance(value, ast.Attribute) and value.attr == "request":
                return isinstance(value.value, ast.Name) and value.value.id == "urllib"
            if isinstance(value, ast.Name) and value.id in {"request", "urllib_request"}:
                return True
        if isinstance(func, ast.Name) and func.id == "urlopen":
            return True
        return False

    def _has_timeout_kw(self, node: ast.Call) -> bool:
        return any(kw.arg == "timeout" and not isinstance(kw.value, ast.Constant) or kw.arg == "timeout" for kw in node.keywords)
