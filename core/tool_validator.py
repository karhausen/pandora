from __future__ import annotations

from .security import ToolSecurityValidator
from .models import ToolSpec


class ToolValidator:
    def __init__(self):
        self.security = ToolSecurityValidator()

    def validate(self, spec: ToolSpec) -> dict:
        errors = self.security.validate_code(spec.code)
        if not spec.id:
            errors.append("Missing tool id")
        if "TOOL_META" not in spec.code:
            errors.append("Missing TOOL_META")
        if "def run(" not in spec.code:
            errors.append("Missing run(payload) function")
        return {"valid": not errors, "errors": errors}
