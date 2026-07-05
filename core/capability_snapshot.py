from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry


@dataclass
class CapabilitySnapshot:
    """Compact, LLM-readable inventory of Pandora's current capabilities.

    This snapshot is descriptive only. It must not inspect the user's wording and
    must not decide routing. The LLM receives this inventory and proposes the
    next action; Pandora validates that proposal before execution.
    """

    tools: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    knowledge_sources: list[dict[str, Any]] = field(default_factory=list)
    memory_sources: list[dict[str, Any]] = field(default_factory=list)
    workflows: list[dict[str, Any]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "tools": self.tools,
            "skills": self.skills,
            "knowledge_sources": self.knowledge_sources,
            "memory_sources": self.memory_sources,
            "workflows": self.workflows,
            "rules": self.rules,
        }


class CapabilitySnapshotBuilder:
    """Builds a capability inventory without keyword-based request routing."""

    def __init__(self, tool_registry: ToolRegistry | None = None, skill_registry: SkillRegistry | None = None):
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or SkillRegistry()

    def build(self) -> CapabilitySnapshot:
        self.tool_registry.discover()
        self.skill_registry.discover()
        tools = []
        for tool in self.tool_registry.list():
            tools.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "status": str(tool.status.value if hasattr(tool.status, "value") else tool.status),
                "security_level": str(tool.security_level.value if hasattr(tool.security_level, "value") else tool.security_level),
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            })
        skills = []
        for skill in self.skill_registry.list():
            skills.append({
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "status": str(skill.status.value if hasattr(skill.status, "value") else skill.status),
                "security_level": str(skill.security_level.value if hasattr(skill.security_level, "value") else skill.security_level),
                "required_tools": skill.required_tools,
            })
        return CapabilitySnapshot(
            tools=tools,
            skills=skills,
            knowledge_sources=[
                {"id": "user_knowledge_base", "description": "Curated Pandora user knowledge files with governance metadata."},
                {"id": "obsidian_vault", "description": "Indexed local Obsidian vault, policy-gated before use."},
            ],
            memory_sources=[
                {"id": "conversation_memory", "description": "Stored conversation facts and session summaries."},
                {"id": "working_memory", "description": "Short-term task/session context."},
            ],
            workflows=[
                {"id": "planner_worker", "description": "Plan and execute existing approved tools/skills."},
                {"id": "tool_factory", "description": "Create a reviewable proposal for a missing tool/capability."},
                {"id": "llm_chat", "description": "Answer with LLM using approved memory/knowledge context."},
            ],
            rules=[
                "No keyword routing. Never select a route because a word occurs in the user request.",
                "The LLM proposes intent and needed capabilities; Pandora validates permissions and availability.",
                "The LLM never directly reads files, executes tools, or writes code without Pandora approval.",
                "If the semantic decision is unavailable, do not select an arbitrary tool fallback.",
            ],
        )
