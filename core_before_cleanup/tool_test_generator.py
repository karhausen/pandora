from __future__ import annotations

from .models import ToolSpec


class ToolTestGenerator:
    """Generic ToolSpec -> pytest generator.

    No capability-specific branches are allowed here. Tests verify only the
    generated tool contract. Semantic correctness belongs to ToolDesign test
    cases and the quality gate when an LLM-generated implementation exists.
    """

    def generate_test(self, spec: ToolSpec) -> str:
        tool_id = spec.id
        output_keys = list((spec.output_schema or {"result": "str"}).keys())
        sample_payload = self._sample_payload(spec.input_schema or {})
        return f'''from generated_tools.{tool_id} import run


def test_{tool_id}_contract():
    result = run({sample_payload!r})
    assert isinstance(result, dict)
    for key in {output_keys!r}:
        assert key in result
'''

    def _sample_payload(self, schema: dict) -> dict:
        return {str(key): self._sample_value(str(type_name)) for key, type_name in schema.items()}

    def _sample_value(self, type_name: str):
        t = str(type_name).lower()
        if t in {"str", "string", "text"}:
            return "sample text"
        if t in {"int", "integer"}:
            return 3
        if t in {"float", "number", "double"}:
            return 3.0
        if t in {"bool", "boolean"}:
            return True
        if t in {"list", "array"}:
            return [1, 2, 3]
        if t in {"dict", "object", "json"}:
            return {"sample": True}
        return None
