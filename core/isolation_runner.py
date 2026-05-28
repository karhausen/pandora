from __future__ import annotations

import json
import uuid
from pathlib import Path

from .config import ROOT_DIR, SANDBOX_RUNS_DIR
from .models import ExecutionPolicy, SandboxResult
from .process_guard import ProcessGuard


class IsolationRunner:
    def __init__(self):
        SANDBOX_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.guard = ProcessGuard()

    def run_tool_isolated(self, tool_id: str, module: str, function: str, payload: dict, policy: ExecutionPolicy) -> SandboxResult:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = SANDBOX_RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        payload_file = run_dir / "payload.json"
        result_file = run_dir / "result.json"
        script_file = run_dir / "runner.py"

        payload_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        script_file.write_text(self._runner_script(module, function, payload_file, result_file), encoding="utf-8")

        proc = self.guard.run_python(script_file, cwd=ROOT_DIR, timeout=policy.timeout)

        if result_file.exists():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
            except Exception as exc:
                data = {"success": False, "error": f"Invalid result JSON: {exc}"}
        else:
            data = {"success": False, "error": proc.get("error") or proc.get("stderr") or "No result produced."}

        return SandboxResult(
            success=bool(proc.get("success") and data.get("success")),
            tool_id=tool_id,
            output=data.get("output"),
            error=data.get("error"),
            execution_time=float(proc.get("execution_time") or 0.0),
            policy=policy.name.value,
            isolated=True,
            returncode=proc.get("returncode"),
        )

    def _runner_script(self, module: str, function: str, payload_file: Path, result_file: Path) -> str:
        # Keep runner intentionally tiny. It imports the tool and writes a normalized result.
        return f'''from __future__ import annotations
import importlib
import json
import sys
import traceback

sys.path.insert(0, r"{ROOT_DIR}")
payload_file = r"{payload_file}"
result_file = r"{result_file}"

try:
    payload = json.loads(open(payload_file, "r", encoding="utf-8").read())
    module = importlib.import_module("{module}")
    fn = getattr(module, "{function}")
    output = fn(payload)
    result = {{"success": True, "output": output, "error": None}}
except Exception as exc:
    result = {{"success": False, "output": None, "error": f"{{type(exc).__name__}}: {{exc}}", "traceback": traceback.format_exc()}}

open(result_file, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False))
'''
