from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .models import ToolDesign
from .tool_review_agent import ToolReviewAgent


class ToolValidator:
    def __init__(self):
        self.reviewer = ToolReviewAgent()

    def static_review(self, code: str, design: ToolDesign | dict[str, Any] | None = None) -> dict:
        return self.reviewer.review(code, design=design)

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
