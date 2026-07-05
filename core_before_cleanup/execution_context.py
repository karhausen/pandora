from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExecutionContext:
    task: str
    analysis: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    used_tools: list[str] = field(default_factory=list)
    used_skills: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
