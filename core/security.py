from __future__ import annotations
import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {"subprocess","socket","ctypes","multiprocessing","shutil","requests","urllib","httpx","ftplib","telnetlib"}
FORBIDDEN_CALLS = {"eval","exec","compile","__import__","open"}
FORBIDDEN_ATTR_CALLS = {("os","system"),("os","popen"),("shutil","rmtree")}

class SecurityViolation(Exception):
    pass

class ToolSecurityValidator:
    def validate_code(self, code: str) -> list[str]:
        errors: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"SyntaxError: {exc}"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                        errors.append(f"Forbidden import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    errors.append(f"Forbidden import: {node.module}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                    errors.append(f"Forbidden call: {node.func.id}")
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    pair = (node.func.value.id, node.func.attr)
                    if pair in FORBIDDEN_ATTR_CALLS:
                        errors.append(f"Forbidden call: {pair[0]}.{pair[1]}")
        return errors
    def validate_target_path(self, target: Path, allowed_root: Path) -> None:
        target_resolved = target.resolve()
        root_resolved = allowed_root.resolve()
        if root_resolved not in target_resolved.parents and target_resolved != root_resolved:
            raise SecurityViolation(f"Path outside allowed root: {target}")
