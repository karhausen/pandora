from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

from .models import ToolDesign, ToolSpec


class ToolQualityGate:
    """Semantic validation for generated tools.

    Static review asks: "Is this code allowed?"
    Pytest asks: "Do generated tests pass?"
    The quality gate asks: "Does the tool contract match the design/schema?"
    """

    def validate(
        self,
        proposal_dir: Path,
        tool_id: str,
        design: ToolDesign | ToolSpec | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []
        checks: dict[str, Any] = {}
        design_data = self._design_dict(design)
        expected_output_schema = self._schema(design_data.get("output_schema"))
        expected_input_schema = self._schema(design_data.get("input_schema"))
        test_cases = list(design_data.get("test_cases") or [])

        try:
            module = self._load_module(proposal_dir, tool_id)
        except Exception as exc:
            return {
                "ok": False,
                "risk": "HIGH",
                "issues": [f"ImportError: {type(exc).__name__}: {exc}"],
                "warnings": warnings,
                "checks": {"importable": False},
            }

        checks["importable"] = True
        meta = getattr(module, "TOOL_META", None)
        run = getattr(module, "run", None)
        if not isinstance(meta, dict):
            issues.append("TOOL_META must be a dict")
            meta = {}
        if not callable(run):
            issues.append("Tool must define callable run(payload)")

        meta_output_schema = self._schema(meta.get("output_schema"))
        meta_input_schema = self._schema(meta.get("input_schema"))
        checks["meta_output_schema"] = meta_output_schema
        checks["expected_output_schema"] = expected_output_schema

        if expected_output_schema and not meta_output_schema:
            issues.append("TOOL_META.output_schema is missing")
        for key in expected_output_schema:
            if key not in meta_output_schema:
                issues.append(f"TOOL_META.output_schema missing key: {key}")
        for key in expected_input_schema:
            if meta_input_schema and key not in meta_input_schema:
                warnings.append(f"TOOL_META.input_schema missing expected key: {key}")

        case_results: list[dict[str, Any]] = []
        if callable(run):
            if test_cases:
                for idx, case in enumerate(test_cases, start=1):
                    payload = case.get("input", {}) if isinstance(case, dict) else {}
                    expected = case.get("expected", {}) if isinstance(case, dict) else {}
                    case_results.append(self._run_case(run, payload, expected, expected_output_schema, idx))
            elif expected_output_schema:
                sample_payload = self._sample_payload(expected_input_schema)
                case_results.append(self._run_case(run, sample_payload, {}, expected_output_schema, 1))

        for case_result in case_results:
            if not case_result.get("ok"):
                issues.extend(case_result.get("issues", []))
        checks["cases"] = case_results

        issues = sorted(set(issues))
        warnings = sorted(set(warnings))
        return {
            "ok": not issues,
            "risk": "HIGH" if issues else ("MEDIUM" if warnings else "LOW"),
            "issues": issues,
            "warnings": warnings,
            "checks": checks,
        }

    def _run_case(
        self,
        run,
        payload: dict[str, Any],
        expected: dict[str, Any],
        output_schema: dict[str, str],
        idx: int,
    ) -> dict[str, Any]:
        issues: list[str] = []
        try:
            output = run(payload)
        except Exception as exc:
            return {
                "case": idx,
                "ok": False,
                "payload": payload,
                "issues": [f"Case {idx} raised {type(exc).__name__}: {exc}"],
            }

        if not isinstance(output, dict):
            return {"case": idx, "ok": False, "payload": payload, "output": repr(output), "issues": [f"Case {idx} output is not a dict"]}

        for key, type_name in output_schema.items():
            if key not in output:
                issues.append(f"Case {idx} output missing schema key: {key}")
            elif not self._matches_type(output[key], type_name):
                issues.append(f"Case {idx} output key {key} has wrong type: expected {type_name}, got {type(output[key]).__name__}")

        for key, expected_value in expected.items():
            if key not in output:
                issues.append(f"Case {idx} output missing expected key: {key}")
            elif output[key] != expected_value:
                issues.append(f"Case {idx} output mismatch for {key}: expected {expected_value!r}, got {output[key]!r}")

        return {"case": idx, "ok": not issues, "payload": payload, "expected": expected, "output": output, "issues": issues}

    def _load_module(self, proposal_dir: Path, tool_id: str):
        path = proposal_dir / "generated_tools" / f"{tool_id}.py"
        if not path.exists():
            raise FileNotFoundError(path)
        module_name = f"pandora_quality_gate_{tool_id}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        sys.path.insert(0, str(proposal_dir))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
        return module

    def _design_dict(self, design: ToolDesign | ToolSpec | dict[str, Any] | None) -> dict[str, Any]:
        if design is None:
            return {}
        if hasattr(design, "model_dump"):
            return design.model_dump(mode="json")
        return dict(design)

    def _schema(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v).lower() for k, v in value.items()}

    def _matches_type(self, value: Any, type_name: str) -> bool:
        t = type_name.lower()
        if t in {"str", "string", "text"}:
            return isinstance(value, str)
        if t in {"int", "integer"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if t in {"float", "number", "double"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t in {"bool", "boolean"}:
            return isinstance(value, bool)
        if t in {"dict", "object", "json"}:
            return isinstance(value, dict)
        if t in {"list", "array"}:
            return isinstance(value, list)
        return True

    def _sample_payload(self, input_schema: dict[str, str]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, type_name in input_schema.items():
            t = type_name.lower()
            if t in {"str", "string", "text"}:
                payload[key] = "eins zwei drei"
            elif t in {"int", "integer"}:
                payload[key] = 3
            elif t in {"float", "number", "double"}:
                payload[key] = 3.0
            elif t in {"bool", "boolean"}:
                payload[key] = True
            elif t in {"list", "array"}:
                payload[key] = []
            else:
                payload[key] = "test"
        return payload
