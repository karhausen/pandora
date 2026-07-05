from __future__ import annotations

from .models import ToolSpec


class ToolCodePrompt:
    def build(self, spec: ToolSpec, previous_error: str | None = None) -> str:
        repair = f"\nPrevious error to fix: {previous_error}\n" if previous_error else ""
        return f'''Generate one safe Python tool module.

Rules:
- Return only Python code.
- No markdown fences.
- Define TOOL_META.
- Define run(payload: dict) -> dict.
- No network.
- No shell.
- No file access.
- No subprocess.
- Keep code small and deterministic.

Tool spec:
id={spec.id}
name={spec.name}
description={spec.description}
input_schema={spec.input_schema}
output_schema={spec.output_schema}
{repair}
'''
