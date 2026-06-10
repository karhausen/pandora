from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_graph import CapabilityGraphService


@dataclass
class CapabilityRegistry:
    """Read-only registry facade for capability nodes."""

    graph: CapabilityGraphService | None = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = CapabilityGraphService()

    def list(self, *, query: str | None = None, limit: int = 200) -> dict[str, Any]:
        return self.graph.list_capabilities(query=query, limit=limit)

    def get(self, capability: str) -> dict[str, Any]:
        return self.graph.show_capability(capability)
