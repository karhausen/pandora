from __future__ import annotations
import subprocess
import sys
from pathlib import Path

class RegressionRunner:
    def run_compileall(self, root: Path, timeout: int = 30) -> dict:
        proc = subprocess.run([sys.executable, "-m", "compileall", "-q", str(root)], cwd=root, text=True, capture_output=True, timeout=timeout)
        return {"name": "compileall", "success": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def run_all(self, root: Path) -> dict:
        # MVP 9B keeps validation intentionally lightweight to avoid recursive pytest-in-pytest loops.
        try:
            result = self.run_compileall(root)
        except Exception as exc:
            result = {"name": "compileall", "success": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"success": bool(result.get("success")), "results": [result]}
