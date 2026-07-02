from __future__ import annotations

import re
from .models import SecurityLevel, ToolSpec


class ToolGenerator:
    """Generic ToolSpec -> Python module generator.

    This generator is deliberately capability-agnostic. It must not contain
    domain branches such as word_count, prime numbers, weather, stocks, etc.

    Architecture:
        LLM/Design Agent understands the requested capability and creates a
        ToolDesign/ToolSpec contract.
        Python turns that contract into a safe module scaffold and validates it.

    If no LLM-generated implementation is available, this generator creates a
    schema-safe contract scaffold. It is not allowed to fake domain knowledge.
    """

    def build_spec(self, capability: str) -> ToolSpec:
        tool_id = self._safe_id(capability)
        return ToolSpec(
            id=tool_id,
            name=tool_id.replace("_", " ").title(),
            description=f"Generated candidate tool for capability: {capability}",
            capability=capability,
            input_schema={"text": "str"},
            output_schema={"result": "str"},
            security_level=SecurityLevel.SAFE,
        )

    def generate_code(self, spec: ToolSpec) -> str:
        module_id = self._safe_id(spec.id)
        input_schema = {str(k): str(v) for k, v in (spec.input_schema or {}).items()}
        output_schema = {str(k): str(v) for k, v in (spec.output_schema or {}).items()} or {"result": "str"}
        output_lines = []
        for key, type_name in output_schema.items():
            output_lines.append(f"        {key!r}: _default_value({type_name!r}),")
        output_block = "\n".join(output_lines)
        return f'''TOOL_META = {{
    "id": "{module_id}",
    "name": {spec.name!r},
    "description": {spec.description!r},
    "version": "0.1.0",
    "input_schema": {input_schema!r},
    "output_schema": {output_schema!r},
    "security_level": "{spec.security_level.value if hasattr(spec.security_level, 'value') else spec.security_level}",
    "status": "ACTIVE",
    "module": "generated_tools.{module_id}",
    "function": "run",
}}


def _default_value(type_name: str):
    t = str(type_name).lower()
    if t in {{"str", "string", "text"}}:
        return ""
    if t in {{"int", "integer"}}:
        return 0
    if t in {{"float", "number", "double"}}:
        return 0.0
    if t in {{"bool", "boolean"}}:
        return False
    if t in {{"list", "array"}}:
        return []
    if t in {{"dict", "object", "json"}}:
        return {{}}
    return None


def _validate_payload(payload: dict) -> dict:
    if payload is None:
        return {{}}
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    return payload


def run(payload: dict) -> dict:
    """Schema-safe generic scaffold.

    This fallback intentionally does not implement domain-specific behavior.
    Real behavior must come from an LLM-generated implementation based on the
    ToolDesign contract and must pass validation before activation.
    """
    _validate_payload(payload)
    return {{
{output_block}
    }}
'''

    def _safe_id(self, capability: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(capability).strip().lower()).strip("_")
        if not value or not re.match(r"^[a-zA-Z_]", value):
            value = f"tool_{value or 'generated'}"
        return value
