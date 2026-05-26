from __future__ import annotations

import re
from .models import ToolSpec


class ToolGenerator:
    def generate(self, capability: str) -> ToolSpec:
        if capability == "csv_processing":
            return self._csv_reader()
        return self._generic_text_tool(capability)

    def _safe_id(self, text: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")

    def _csv_reader(self) -> ToolSpec:
        code = '''from __future__ import annotations
import csv
from pathlib import Path

TOOL_META = {
    "id": "csv_reader",
    "name": "CSV Reader",
    "description": "Reads a CSV file from an allowed path and returns rows plus basic numeric summaries.",
    "version": "0.1.0",
    "input_schema": {"path": "str"},
    "output_schema": {"rows": "list", "summary": "dict"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "tools.csv_reader",
    "function": "run"
}

def run(payload: dict) -> dict:
    path = Path(payload["path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported")

    with path.open("r", encoding=payload.get("encoding", "utf-8"), newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    summary = {"row_count": len(rows), "columns": reader.fieldnames or []}
    numeric = {}
    for col in summary["columns"]:
        values = []
        for row in rows:
            try:
                values.append(float(row[col]))
            except Exception:
                pass
        if values:
            numeric[col] = {
                "count": len(values),
                "sum": sum(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    summary["numeric"] = numeric
    return {"rows": rows[:100], "summary": summary}
'''
        return ToolSpec(
            id="csv_reader",
            name="CSV Reader",
            description="Reads CSV files and creates simple summaries.",
            capability="csv_processing",
            input_schema={"path": "str"},
            output_schema={"rows": "list", "summary": "dict"},
            code=code,
            tests=[{"payload": {"path": "__SAMPLE_CSV__"}, "expect_success": True}],
        )

    def _generic_text_tool(self, capability: str) -> ToolSpec:
        tool_id = self._safe_id(capability) + "_tool"
        code = f'''TOOL_META = {{
    "id": "{tool_id}",
    "name": "{capability} Tool",
    "description": "Generated placeholder tool for capability: {capability}",
    "version": "0.1.0",
    "input_schema": {{"text": "str"}},
    "output_schema": {{"result": "str"}},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "tools.{tool_id}",
    "function": "run"
}}

def run(payload: dict) -> dict:
    return {{"result": payload.get("text", "")}}
'''
        return ToolSpec(
            id=tool_id,
            name=f"{capability} Tool",
            description=f"Generated placeholder tool for {capability}",
            capability=capability,
            input_schema={"text": "str"},
            output_schema={"result": "str"},
            code=code,
            tests=[{"payload": {"text": "hello"}, "expect_success": True}],
        )
