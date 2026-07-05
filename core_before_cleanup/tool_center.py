from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tool_lifecycle_manager import ToolLifecycleManager
from .tool_registry import ToolRegistry


@dataclass
class ToolCenterService:
    """Read-only first GUI service for installed tool visibility and safe lifecycle actions."""

    registry: ToolRegistry | None = None
    lifecycle: ToolLifecycleManager | None = None

    def __post_init__(self) -> None:
        self.registry = self.registry or ToolRegistry()
        self.lifecycle = self.lifecycle or ToolLifecycleManager(self.registry)

    def dashboard(self) -> dict[str, Any]:
        tools = self._tool_cards()
        counts: dict[str, int] = {}
        security: dict[str, int] = {}
        for tool in tools:
            counts[tool["status"]] = counts.get(tool["status"], 0) + 1
            security[tool["security_level"]] = security.get(tool["security_level"], 0) + 1
        return {
            "tool_count": len(tools),
            "status_counts": counts,
            "security_counts": security,
            "tools": tools,
        }

    def list_tools(self, status: str | None = None, include_stats: bool = True) -> dict[str, Any]:
        tools = self._tool_cards(include_stats=include_stats)
        if status:
            wanted = status.upper()
            tools = [tool for tool in tools if tool["status"].upper() == wanted]
        return {"count": len(tools), "tools": tools}

    def show_tool(self, tool_id: str) -> dict[str, Any]:
        info = self.lifecycle.info(tool_id)
        if not info.success:
            return {"found": False, "tool_id": tool_id, "error": info.error or "Tool not found"}
        return {"found": True, "tool": info.tool, "stats": info.stats}

    def set_tool_status(self, tool_id: str, action: str) -> dict[str, Any]:
        normalized = action.strip().lower().replace("_", "-")
        if normalized == "enable":
            result = self.lifecycle.enable(tool_id)
        elif normalized == "disable":
            result = self.lifecycle.disable(tool_id)
        elif normalized == "deprecate":
            result = self.lifecycle.deprecate(tool_id)
        else:
            raise ValueError("Unsupported tool action. Allowed: enable, disable, deprecate")
        return result.model_dump(mode="json")

    def stats(self, tool_id: str | None = None) -> dict[str, Any]:
        return {"stats": self.lifecycle.stats(tool_id)}

    def _tool_cards(self, include_stats: bool = True) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for meta in sorted(self.registry.list(), key=lambda item: item.id):
            stats = self.lifecycle.stats(meta.id) if include_stats else None
            executions = int((stats or {}).get("executions", 0) or 0)
            successes = int((stats or {}).get("successes", 0) or 0)
            success_rate = round(successes / executions, 4) if executions else None
            cards.append(
                {
                    "id": meta.id,
                    "name": meta.name,
                    "description": meta.description,
                    "version": meta.version,
                    "status": meta.status.value,
                    "security_level": meta.security_level.value,
                    "module": meta.module,
                    "function": meta.function,
                    "aliases": list(meta.aliases or []),
                    "installed_from": meta.installed_from,
                    "input_schema": meta.input_schema,
                    "output_schema": meta.output_schema,
                    "stats": stats,
                    "success_rate": success_rate,
                }
            )
        return cards
