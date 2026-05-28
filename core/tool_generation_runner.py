from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class ToolGenerationRunner:
    def run_pytest(self, proposal_dir: Path, timeout: int = 30) -> dict:
        try:
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
        except subprocess.TimeoutExpired:
            return {"success": False, "returncode": None, "stdout": "", "stderr": f"Timeout after {timeout}s"}
