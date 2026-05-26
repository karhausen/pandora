from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


class SandboxRunner:
    def __init__(self, timeout: int = 2):
        self.timeout = timeout

    def run_command(self, version_path: Path, args: list[str]) -> dict:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, *args],
                cwd=version_path,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "execution_time": time.perf_counter() - start,
            }
        except Exception as exc:
            return {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
                "execution_time": time.perf_counter() - start,
            }

    def run_heartbeat(self, version_path: Path) -> dict:
        return self.run_command(version_path, ["main.py", "heartbeat"])

    def run_smoke_tests(self, version_path: Path) -> dict:
        tests = [
            ["main.py", "status"],
            ["main.py", "heartbeat"],
            ["main.py", "run-tool", "echo", "--input", "sandbox"],
        ]
        results = []
        for args in tests:
            results.append({"command": args, "result": self.run_command(version_path, args)})
        success = all(r["result"]["success"] for r in results)
        return {"success": success, "results": results}

    def write_results(self, version_path: Path, filename: str, payload: dict) -> None:
        (version_path / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
