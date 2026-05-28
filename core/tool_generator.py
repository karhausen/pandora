from __future__ import annotations

import re
from .models import SecurityLevel, ToolSpec


class ToolGenerator:
    def build_spec(self, capability: str) -> ToolSpec:
        tool_id = self._safe_id(capability)
        catalog = {
            "json_pretty": ToolSpec(
                id="json_pretty",
                name="JSON Pretty Printer",
                description="Formats JSON text with indentation.",
                capability="json_pretty",
                input_schema={"text": "str"},
                output_schema={"text": "str"},
                security_level=SecurityLevel.SAFE,
            ),
            "text_reverse": ToolSpec(
                id="text_reverse",
                name="Text Reverse",
                description="Reverses input text.",
                capability="text_reverse",
                input_schema={"text": "str"},
                output_schema={"text": "str"},
                security_level=SecurityLevel.SAFE,
            ),
            "word_count": ToolSpec(
                id="word_count",
                name="Word Count",
                description="Counts words in input text.",
                capability="word_count",
                input_schema={"text": "str"},
                output_schema={"count": "int"},
                security_level=SecurityLevel.SAFE,
            ),
            "timestamp": ToolSpec(
                id="timestamp",
                name="Timestamp",
                description="Returns current UTC timestamp.",
                capability="timestamp",
                input_schema={},
                output_schema={"timestamp": "str"},
                security_level=SecurityLevel.SAFE,
            ),
        }
        return catalog.get(tool_id, ToolSpec(
            id=tool_id,
            name=tool_id.replace("_", " ").title(),
            description=f"Generated candidate tool for capability: {capability}",
            capability=capability,
            input_schema={"text": "str"},
            output_schema={"text": "str"},
            security_level=SecurityLevel.SAFE,
        ))

    def generate_code(self, spec: ToolSpec) -> str:
        if spec.id == "json_pretty":
            return '''import json

TOOL_META = {
    "id": "json_pretty",
    "name": "JSON Pretty Printer",
    "description": "Formats JSON text with indentation.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"text": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.json_pretty",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    data = json.loads(text)
    return {"text": json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)}
'''
        if spec.id == "text_reverse":
            return '''TOOL_META = {
    "id": "text_reverse",
    "name": "Text Reverse",
    "description": "Reverses input text.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"text": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.text_reverse",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    return {"text": str(text)[::-1]}
'''
        if spec.id == "word_count":
            return '''TOOL_META = {
    "id": "word_count",
    "name": "Word Count",
    "description": "Counts words in input text.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"count": "int"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.word_count",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    words = [w for w in str(text).split() if w.strip()]
    return {"count": len(words)}
'''
        if spec.id == "timestamp":
            return '''from datetime import datetime, UTC

TOOL_META = {
    "id": "timestamp",
    "name": "Timestamp",
    "description": "Returns current UTC timestamp.",
    "version": "0.1.0",
    "input_schema": {},
    "output_schema": {"timestamp": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.timestamp",
    "function": "run",
}

def run(payload: dict) -> dict:
    return {"timestamp": datetime.now(UTC).isoformat()}
'''
        return f'''TOOL_META = {{
    "id": "{spec.id}",
    "name": "{spec.name}",
    "description": "{spec.description}",
    "version": "0.1.0",
    "input_schema": {spec.input_schema!r},
    "output_schema": {spec.output_schema!r},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.{spec.id}",
    "function": "run",
}}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    return {{"text": str(text)}}
'''

    def _safe_id(self, capability: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", capability.strip().lower()).strip("_")
        return value or "generated_tool"
