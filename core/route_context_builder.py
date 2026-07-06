from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import os


@dataclass
class BuiltRouteContext:
    """Normalized context assembled from one or more executed routes."""

    context_text: str
    sources: list[str] = field(default_factory=list)
    source_count: int = 0
    used_chars: int = 0
    truncated: bool = False
    route_kinds: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "sources": self.sources,
            "used_chars": self.used_chars,
            "truncated": self.truncated,
            "route_kinds": self.route_kinds,
        }


class RouteContextBuilder:
    """Builds the final LLM context from executed route results.

    The builder does not decide which route to use. It only normalizes and
    combines context that has already been requested by the LLM and dispatched
    by Python. This keeps the router decision-free while making the final
    answer prompt more deterministic and auditable.
    """

    def __init__(self, max_total_chars: int | None = None):
        if max_total_chars is None:
            max_total_chars = int(os.getenv("PANDORA_ROUTE_CONTEXT_MAX_CHARS", "12000"))
        self.max_total_chars = max(1000, int(max_total_chars))

    def build(self, route_results: list[dict[str, Any]], memory_summary: str = "") -> BuiltRouteContext:
        parts: list[str] = []
        sources: list[str] = []
        route_kinds: list[str] = []
        truncated = False

        if memory_summary:
            parts.append("Gesprächsgedächtnis:\n" + memory_summary.strip())

        for route_result in route_results:
            kind = str(route_result.get("kind") or route_result.get("route") or "context")
            if kind not in route_kinds:
                route_kinds.append(kind)

            context_text = str(route_result.get("context_text") or "").strip()
            if context_text:
                parts.append(f"Von Pandora bereitgestellter Kontext ({kind}):\n{context_text}")

            for label in self._source_labels(route_result.get("sources") or []):
                if label not in sources:
                    sources.append(label)

        if sources:
            parts.append("Verwendbare Quellen:\n" + "\n".join(f"{idx}. {label}" for idx, label in enumerate(sources, start=1)))

        full_text = "\n\n".join(part for part in parts if part.strip())
        if len(full_text) > self.max_total_chars:
            full_text = full_text[: self.max_total_chars].rstrip() + "\n\n[Kontext gekürzt: Zeichenbudget erreicht]"
            truncated = True

        return BuiltRouteContext(
            context_text=full_text,
            sources=sources,
            source_count=len(sources),
            used_chars=len(full_text),
            truncated=truncated,
            route_kinds=route_kinds,
        )

    def _source_labels(self, sources: list[Any]) -> list[str]:
        labels: list[str] = []
        for src in sources:
            if isinstance(src, dict):
                label = src.get("relative_path") or src.get("source_id") or src.get("title") or src.get("source_type")
            else:
                label = str(src)
            if label:
                labels.append(str(label))
        return labels
