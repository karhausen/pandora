from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


class ToolValidator:
    FORBIDDEN_IMPORTS = {"subprocess", "socket", "ctypes", "multiprocessing", "shutil", "requests", "urllib", "httpx"}
    FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open"}

    def static_review(self, code: str) -> dict:
        issues: list[str] = []
        warnings: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"ok": False, "risk": "HIGH", "issues": [f"SyntaxError: {exc}"], "warnings": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.FORBIDDEN_IMPORTS:
                        issues.append(f"Forbidden import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in self.FORBIDDEN_IMPORTS:
                    issues.append(f"Forbidden import: {node.module}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_CALLS:
                    issues.append(f"Forbidden call: {node.func.id}")

        risk = "HIGH" if issues else ("MEDIUM" if warnings else "LOW")
        return {"ok": not issues, "risk": risk, "issues": issues, "warnings": warnings}

    def run_tests(self, proposal_dir: Path, timeout: int = 30) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=proposal_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
