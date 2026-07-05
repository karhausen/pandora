from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capability_model import CapabilityRecord
from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry


@dataclass
class CapabilitySnapshot:
    """Compact, LLM-readable inventory of Pandora's current capabilities.

    This snapshot is descriptive only. It must not inspect the user's wording and
    must not decide routing. The LLM receives this inventory and proposes the
    next action; Pandora validates that proposal before execution.
    """

    capabilities: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    knowledge_sources: list[dict[str, Any]] = field(default_factory=list)
    memory_sources: list[dict[str, Any]] = field(default_factory=list)
    workflows: list[dict[str, Any]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
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
        capabilities: list[dict[str, Any]] = []
        tools = []
        for tool in self.tool_registry.list():
            status = str(tool.status.value if hasattr(tool.status, "value") else tool.status)
            security_level = str(tool.security_level.value if hasattr(tool.security_level, "value") else tool.security_level)
            record = CapabilityRecord(
                id=f"tool:{tool.id}",
                name=tool.name,
                kind="tool",
                description=tool.description,
                status=status,
                security_level=security_level,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                permissions=["execute_tool"],
                provider="tool_registry",
                implementation_ref=tool.id,
                reliability="installed",
            ).model_dump()
            capabilities.append(record)
            tools.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "status": status,
                "security_level": security_level,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "capability_id": record["id"],
            })
        skills = []
        for skill in self.skill_registry.list():
            status = str(skill.status.value if hasattr(skill.status, "value") else skill.status)
            security_level = str(skill.security_level.value if hasattr(skill.security_level, "value") else skill.security_level)
            record = CapabilityRecord(
                id=f"skill:{skill.id}",
                name=skill.name,
                kind="skill",
                description=skill.description,
                status=status,
                security_level=security_level,
                required_capabilities=[f"tool:{tool_id}" for tool_id in skill.required_tools],
                permissions=["run_skill"],
                provider="skill_registry",
                implementation_ref=skill.id,
                reliability="installed",
            ).model_dump()
            capabilities.append(record)
            skills.append({
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "status": status,
                "security_level": security_level,
                "required_tools": skill.required_tools,
                "capability_id": record["id"],
            })
        knowledge_sources = [
            {"id": "user_knowledge_base", "description": "Curated Pandora user knowledge files with governance metadata."},
            {"id": "obsidian_vault", "description": "Indexed local Obsidian vault, policy-gated before use."},
        ]
        memory_sources = [
            {"id": "conversation_memory", "description": "Stored conversation facts and session summaries."},
            {"id": "working_memory", "description": "Short-term task/session context."},
        ]
        workflows = [
            {"id": "planner_worker", "description": "Plan and execute existing approved tools/skills."},
            {"id": "python_task_execution", "description": "Use approved local Python/tool execution for one-time deterministic computations when no persistent new capability is required."},
            {"id": "tool_factory", "description": "Create a reviewable proposal for a missing persistent tool/capability only after existing capabilities are insufficient."},
            {"id": "llm_chat", "description": "Answer with LLM using approved memory/knowledge context."},
        ]
        for source in knowledge_sources:
            capabilities.append(CapabilityRecord(
                id=f"knowledge:{source['id']}",
                name=source["id"].replace("_", " ").title(),
                kind="knowledge",
                description=source["description"],
                permissions=["read_knowledge"],
                provider=source["id"],
                implementation_ref=source["id"],
                reliability="policy_gated",
            ).model_dump())
        for source in memory_sources:
            capabilities.append(CapabilityRecord(
                id=f"memory:{source['id']}",
                name=source["id"].replace("_", " ").title(),
                kind="memory",
                description=source["description"],
                permissions=["read_memory"],
                provider=source["id"],
                implementation_ref=source["id"],
                reliability="policy_gated",
            ).model_dump())
        for workflow in workflows:
            capabilities.append(CapabilityRecord(
                id=f"workflow:{workflow['id']}",
                name=workflow["id"].replace("_", " ").title(),
                kind="workflow",
                description=workflow["description"],
                permissions=["run_workflow"],
                provider="pandora_core",
                implementation_ref=workflow["id"],
                reliability="core",
            ).model_dump())
        return CapabilitySnapshot(
            capabilities=capabilities,
            tools=tools,
            skills=skills,
            knowledge_sources=knowledge_sources,
            memory_sources=memory_sources,
            workflows=workflows,
            rules=[
                "No keyword routing. Never select a route because a word occurs in the user request.",
                "Plan only against neutral CapabilityRecord objects, not hard-coded implementation categories.",
                "The LLM proposes intent and needed capabilities; Pandora validates permissions and availability.",
                "The LLM never directly reads files, executes tools, or writes code without Pandora approval.",
                "Before creating a new capability proposal, first consider direct reasoning, knowledge, memory, existing tools, Python task execution, skills, and workflows.",
                "If the semantic decision is unavailable, do not select an arbitrary tool fallback.",
            ],
        )
