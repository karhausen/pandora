from __future__ import annotations
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
