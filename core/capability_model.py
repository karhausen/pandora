from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityRecord:
    """Unified, LLM-readable description of one Pandora capability.

    A capability can be backed by a tool, skill, knowledge source, memory source,
    workflow, or internal service. The orchestrator plans against this neutral
    model instead of against implementation-specific categories.
    """

    id: str
    name: str
    kind: str
    description: str
    status: str = "available"
    security_level: str = "SAFE"
    input_schema: Any = field(default_factory=dict)
    output_schema: Any = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    reliability: str = "unknown"
    provider: str | None = None
    implementation_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "status": self.status,
            "security_level": self.security_level,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "required_capabilities": self.required_capabilities,
            "permissions": self.permissions,
            "cost": self.cost,
            "reliability": self.reliability,
            "provider": self.provider,
            "implementation_ref": self.implementation_ref,
            "metadata": self.metadata,
        }
