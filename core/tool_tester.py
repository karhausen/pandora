from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from .models import ToolSpec


class ToolTester:
    def run_tests(self, spec: ToolSpec, tools_dir: Path) -> dict:
        sample_csv = None
        with tempfile.TemporaryDirectory() as tmp:
            sample_csv = Path(tmp) / "sample.csv"
            sample_csv.write_text("name,value\na,1\nb,2\n", encoding="utf-8")

            target = tools_dir / f"{spec.id}.py"
            if not target.exists():
                return {"passed": False, "errors": [f"Tool file not found: {target}"]}

            module_name = f"tools.{spec.id}"
            sys.modules.pop(module_name, None)
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                return {"passed": False, "errors": [f"Import failed: {type(exc).__name__}: {exc}"]}

            errors = []
            for test in spec.tests:
                payload = dict(test.get("payload", {}))
                if payload.get("path") == "__SAMPLE_CSV__":
                    payload["path"] = str(sample_csv)
                try:
                    result = module.run(payload)
                    if test.get("expect_success") and result is None:
                        errors.append("Expected result, got None")
                except Exception as exc:
                    if test.get("expect_success", True):
                        errors.append(f"Execution failed: {type(exc).__name__}: {exc}")

            return {"passed": not errors, "errors": errors}
