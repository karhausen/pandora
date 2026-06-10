from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_graph import CapabilityGraphService


@dataclass
class CapabilityRelationshipService:
    """Helper for relationship queries over the persisted capability graph."""

    graph: CapabilityGraphService | None = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = CapabilityGraphService()

    def graph_payload(self) -> dict[str, Any]:
        return self.graph.load_graph()

    def related(self, capability: str) -> dict[str, Any]:
        return self.graph.show_capability(capability)
