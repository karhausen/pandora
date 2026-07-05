from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm_config import LLMConfig
from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry


KNOWN_SOURCE_SPACES = [
    "conversation_memory",
    "long_term_memory",
    "user_knowledge",
    "obsidian_vault",
    "capability_graph",
    "learning_engine",
    "tool_registry",
    "skill_registry",
]


@dataclass
class RequestInterpreter:
    """LLM-assisted semantic request interpretation for Pandora.

    The interpreter does not read files, execute tools or make final decisions.
    It asks the cognitive brain for a structured recommendation about intent,
    relevant source spaces and possible tool/skill needs. Python remains
    responsible for validation, governance and execution.
    """

    llm: LLMRuntime | None = None
    tool_registry: ToolRegistry | None = None
    skill_registry: SkillRegistry | None = None
    llm_config: LLMConfig | None = None
    source_spaces: list[str] = field(default_factory=lambda: list(KNOWN_SOURCE_SPACES))

    def __post_init__(self) -> None:
        self.llm_config = self.llm_config or LLMConfig()
        self.llm = self.llm or LLMRuntime(self.llm_config)
        self.tool_registry = self.tool_registry or ToolRegistry()
        self.skill_registry = self.skill_registry or SkillRegistry()

    def interpret(self, request: str, *, provider_name: str | None = None, model: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
        available_tools = self._available_tools()
        available_skills = self._available_skills()
        prompt = self._build_prompt(request, available_tools=available_tools, available_skills=available_skills)
        system_prompt = (
            "You are Pandora's Request Interpreter. Return ONLY valid JSON. "
            "Do not answer the user request. Do not execute tools. Do not request file contents. "
            "Recommend source spaces and capability needs only. Python validates and acts."
        )
        try:
            response = self.llm.complete(LLMRequest(
                task_type=LLMTaskType.PLANNING,
                prompt=prompt,
                system_prompt=system_prompt,
                context={
                    "task": request,
                    "available_sources": self.source_spaces,
                    "available_tools": available_tools,
                    "available_skills": available_skills,
                },
                provider_name=provider_name,
                model=model,
                expect_json=True,
                timeout=timeout,
            ))
            if response.success and isinstance(response.parsed_json, dict):
                parsed = response.parsed_json
                normalized = self._normalize(parsed, request, available_tools=available_tools, available_skills=available_skills)
                normalized["interpreter"] = {
                    "mode": "llm_recommendation",
                    "provider_name": response.provider_name,
                    "model": response.model,
                    "recovered": response.recovered,
                    "confidence": response.confidence,
                }
                return normalized
            return self._heuristic_interpretation(request, available_tools=available_tools, available_skills=available_skills, error=response.error or "invalid_llm_json")
        except Exception as exc:
            return self._heuristic_interpretation(request, available_tools=available_tools, available_skills=available_skills, error=str(exc))

    def status(self) -> dict[str, Any]:
        return {
            "kind": "request_interpreter_status",
            "ok": True,
            "role": "semantic_recommendation_only",
            "guarantee": "No file access, no tool execution, no final decision.",
            "source_spaces": self.source_spaces,
            "available_tools": self._available_tools(),
            "available_skills": self._available_skills(),
            "pipeline_position": "before_python_orchestrator_and_context_builder",
        }

    def _available_tools(self) -> list[dict[str, Any]]:
        try:
            self.tool_registry.discover()
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "status": str(t.status.value if hasattr(t.status, "value") else t.status),
                    "security_level": str(t.security_level.value if hasattr(t.security_level, "value") else t.security_level),
                    "input_schema": t.input_schema,
                    "output_schema": t.output_schema,
                }
                for t in self.tool_registry.list()
            ]
        except Exception:
            return []

    def _available_skills(self) -> list[dict[str, Any]]:
        try:
            self.skill_registry.discover()
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "status": str(s.status.value if hasattr(s.status, "value") else s.status),
                    "required_tools": s.required_tools,
                }
                for s in self.skill_registry.list()
            ]
        except Exception:
            return []

    def _build_prompt(self, request: str, *, available_tools: list[dict[str, Any]], available_skills: list[dict[str, Any]]) -> str:
        schema = {
            "intent": "short intent name",
            "summary": "short summary",
            "source_spaces": ["one or more available source spaces"],
            "tools": [{"id": "tool id", "reason": "why", "required": False, "available": True}],
            "skills": [{"id": "skill id", "reason": "why", "required": False, "available": True}],
            "capability_gaps": [{"type": "tool|skill|knowledge|core", "name": "gap name", "reason": "why"}],
            "confidence": 0.0,
            "reasoning_summary": "brief rationale",
            "recommended_next_step": "context_lookup|tool_use|tool_factory|answer|clarify|core_review|knowledge_update",
            "safety_notes": ["notes"],
        }
        return (
            "Analyze this Pandora user request semantically. Recommend source spaces, existing tools/skills, "
            "and missing capabilities. Return ONLY JSON matching this schema.\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Available source spaces:\n{json.dumps(self.source_spaces, ensure_ascii=False)}\n\n"
            f"Available tools:\n{json.dumps(available_tools, ensure_ascii=False)}\n\n"
            f"Available skills:\n{json.dumps(available_skills, ensure_ascii=False)}\n\n"
            f"User request:\n{request}"
        )

    def _normalize(self, data: dict[str, Any], request: str, *, available_tools: list[dict[str, Any]], available_skills: list[dict[str, Any]]) -> dict[str, Any]:
        tool_ids = {t.get("id") for t in available_tools}
        skill_ids = {s.get("id") for s in available_skills}
        source_spaces = [s for s in data.get("source_spaces", []) if s in self.source_spaces]
        if not source_spaces:
            source_spaces = self._heuristic_sources(request)
        tools = []
        for item in data.get("tools", []) or []:
            if not isinstance(item, dict):
                continue
            tool_id = str(item.get("id") or "").strip()
            if not tool_id:
                continue
            tools.append({
                "id": tool_id,
                "reason": str(item.get("reason") or ""),
                "required": bool(item.get("required", False)),
                "available": tool_id in tool_ids if item.get("available") is None else bool(item.get("available")),
            })
        skills = []
        for item in data.get("skills", []) or []:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("id") or "").strip()
            if not skill_id:
                continue
            skills.append({
                "id": skill_id,
                "reason": str(item.get("reason") or ""),
                "required": bool(item.get("required", False)),
                "available": skill_id in skill_ids if item.get("available") is None else bool(item.get("available")),
            })
        confidence = self._bounded_float(data.get("confidence"), default=0.65)
        heuristic_intent = self._heuristic_intent(request)
        raw_intent = str(data.get("intent") or heuristic_intent)
        if heuristic_intent in {"knowledge_lookup", "tool_use", "capability_or_build_request"} and raw_intent in {"task_execution", "chat_or_analysis"}:
            raw_intent = heuristic_intent
        raw_summary = str(data.get("summary") or request[:160])
        if raw_summary.lower().startswith("analyze this pandora user request"):
            raw_summary = request[:160]
        return {
            "kind": "request_interpretation",
            "request": request,
            "intent": raw_intent,
            "summary": raw_summary,
            "source_spaces": source_spaces,
            "tools": tools,
            "skills": skills,
            "capability_gaps": self._normalize_gaps(data.get("capability_gaps", [])),
            "confidence": confidence,
            "reasoning_summary": str(data.get("reasoning_summary") or "Semantische Empfehlung für den Python-Orchestrator."),
            "recommended_next_step": str(data.get("recommended_next_step") or self._heuristic_next_step(request, tools)),
            "safety_notes": [str(x) for x in data.get("safety_notes", []) if x],
            "available": {"source_spaces": self.source_spaces, "tool_count": len(available_tools), "skill_count": len(available_skills)},
            "rule": "LLM recommends only; Python validates policies and execution.",
        }

    def _normalize_gaps(self, gaps: Any) -> list[dict[str, str]]:
        normalized = []
        for gap in gaps or []:
            if isinstance(gap, dict):
                normalized.append({
                    "type": str(gap.get("type") or "unknown"),
                    "name": str(gap.get("name") or gap.get("capability") or "unknown"),
                    "reason": str(gap.get("reason") or ""),
                })
            elif isinstance(gap, str):
                normalized.append({"type": "unknown", "name": gap, "reason": "LLM-reported capability gap."})
        return normalized

    def _heuristic_interpretation(self, request: str, *, available_tools: list[dict[str, Any]], available_skills: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
        """Safe fallback when LLM interpretation is unavailable.

        This fallback deliberately avoids inspecting request keywords. It returns
        broad, policy-safe context spaces and never selects a tool, skill, gap,
        or route. Final orchestration is handled by CapabilityOrchestrator.
        """
        return {
            "kind": "request_interpretation",
            "request": request,
            "intent": "semantic_interpretation_unavailable",
            "summary": request[:160],
            "source_spaces": ["conversation_memory", "user_knowledge", "obsidian_vault"],
            "tools": [],
            "skills": [],
            "capability_gaps": [],
            "confidence": 0.2,
            "reasoning_summary": "LLM interpretation unavailable; no keyword fallback was used.",
            "recommended_next_step": "answer",
            "safety_notes": ["No keyword routing. No tool selected by fallback."],
            "available": {"source_spaces": self.source_spaces, "tool_count": len(available_tools), "skill_count": len(available_skills)},
            "interpreter": {"mode": "safe_no_keyword_fallback", "error": error},
            "rule": "LLM recommends only; Python validates policies and execution.",
        }

    def _heuristic_sources(self, request: str) -> list[str]:
        return ["conversation_memory", "user_knowledge", "obsidian_vault"]

    def _heuristic_intent(self, request: str) -> str:
        return "semantic_interpretation_unavailable"

    def _heuristic_next_step(self, request: str, tools: list[dict[str, Any]]) -> str:
        return "answer"

    def _bounded_float(self, value: Any, *, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return default
