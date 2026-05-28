from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


class ProcessGuard:
    def run_python(self, script: Path, cwd: Path, timeout: float) -> dict:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "execution_time": time.perf_counter() - start,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "error": f"Timeout after {timeout}s",
                "execution_time": time.perf_counter() - start,
            }
