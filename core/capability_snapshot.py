from __future__ import annotations

from typing import Any

from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


class CapabilitySnapshotBuilder:
    """Build the single JSON state view used by semantic capability decisions.

    The decision engine must not read scattered internals or apply keyword rules.
    Python only collects facts: installed tools, skills, knowledge/workflow hooks,
    and policy constraints. The LLM interprets the user task against this snapshot.
    """

    def __init__(self, tool_registry: ToolRegistry | None = None, skill_registry: SkillRegistry | None = None):
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()

    def build(self) -> dict[str, Any]:
        try:
            self.tool_registry.discover()
        except Exception:
            pass
        try:
            self.skill_registry.discover()
        except Exception:
            pass
        return {
            "tools": self._tools(),
            "skills": self._skills(),
            "knowledge": self._knowledge(),
            "workflows": self._workflows(),
            "capabilities": self._capabilities(),
            "genome": self._genome_summary(),
            "policies": {
                "llm_understands_user_goal": True,
                "python_collects_facts_only": True,
                "python_validates_references_only": True,
                "no_keyword_or_pattern_capability_decisions": True,
                "no_capability_specific_python_branches": True,
                "never_execute_unrelated_fallback_tools": True,
                "missing_capabilities_require_reviewable_proposal": True,
                "human_approval_required_for_activation": True,
            },
        }

    def _tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        try:
            for tool in self.tool_registry.list():
                result.append({
                    "id": str(tool.id),
                    "name": str(tool.name),
                    "description": str(tool.description),
                    "input_schema": dict(tool.input_schema or {}),
                    "output_schema": dict(tool.output_schema or {}),
                    "status": str(tool.status.value if hasattr(tool.status, "value") else tool.status),
                    "security_level": str(tool.security_level.value if hasattr(tool.security_level, "value") else tool.security_level),
                    "aliases": list(getattr(tool, "aliases", []) or []),
                })
        except Exception:
            return []
        return result

    def _skills(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        try:
            for skill in self.skill_registry.list():
                result.append({
                    "id": str(skill.id),
                    "name": str(skill.name),
                    "description": str(skill.description),
                    "required_tools": list(skill.required_tools or []),
                    "input_schema": dict(getattr(skill, "input_schema", {}) or {}),
                    "output_schema": dict(getattr(skill, "output_schema", {}) or {}),
                    "status": str(skill.status.value if hasattr(skill.status, "value") else skill.status),
                })
        except Exception:
            return []
        return result

    def _knowledge(self) -> list[dict[str, Any]]:
        # Hook point for Knowledge Registry/Index. Empty is a factual snapshot,
        # not a decision. The LLM still sees that no indexed knowledge is exposed.
        return []

    def _workflows(self) -> list[dict[str, Any]]:
        # Hook point for Workflow Registry.
        return []

    def _capabilities(self) -> list[dict[str, Any]]:
        capabilities: list[dict[str, Any]] = []
        for tool in self._tools():
            capabilities.append({
                "source": "tool",
                "source_id": tool["id"],
                "name": tool["id"],
                "description": tool.get("description", ""),
            })
        for skill in self._skills():
            capabilities.append({
                "source": "skill",
                "source_id": skill["id"],
                "name": skill["id"],
                "description": skill.get("description", ""),
            })
        return capabilities

    def _genome_summary(self) -> dict[str, Any]:
        try:
            from .genome.genome_manager import GenomeManager

            status = GenomeManager().status()
            return {
                "available": True,
                "status": status.get("status") or status.get("kind") or "available",
                "version": status.get("version"),
            }
        except Exception:
            return {"available": False}
