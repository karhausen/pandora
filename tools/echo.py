from __future__ import annotations

from typing import Any

METADATA = {
    "id": "builtin.echo",
    "name": "echo",
    "description": "Gibt den Payload unverändert zurück. Nützlich für Executor-, CLI- und Pipeline-Tests.",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object", "properties": {"echo": {"type": "object"}}},
    "safety_level": "low",
    "version": "0.2.0",
    "test_status": "tested",
}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    return {"echo": payload}
