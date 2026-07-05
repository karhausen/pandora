from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .request_interpreter import RequestInterpreter
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry

GAP_TYPES = {"tool", "skill", "knowledge", "core"}


@dataclass
class CapabilityAnalyzer:
    """Diagnoses capability gaps from a Pandora request interpretation.

    The analyzer does not execute tools, generate code, read vault files or alter
    registries. It converts semantic recommendations into a validated Python-side
    diagnosis for the future Python Orchestrator.
    """

    request_interpreter: RequestInterpreter | None = None
    tool_registry: ToolRegistry | None = None
    skill_registry: SkillRegistry | None = None

    def __post_init__(self) -> None:
        self.request_interpreter = self.request_interpreter or RequestInterpreter()
        self.tool_registry = self.tool_registry or ToolRegistry()
        self.skill_registry = self.skill_registry or SkillRegistry()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "capability_analyzer_status",
            "ok": True,
            "role": "gap_diagnosis_only",
            "guarantee": "No tool execution, no code generation, no registry activation, no file access.",
            "input": "request_interpretation or raw request",
            "gap_types": sorted(GAP_TYPES),
            "recommended_actions": [
                "context_lookup",
                "tool_use_review",
                "tool_factory_proposal",
                "skill_proposal",
                "knowledge_update_proposal",
                "core_review_proposal",
                "answer",
                "clarify",
            ],
            "pipeline_position": "after_request_interpreter_before_python_orchestrator",
        }

    def analyze(
        self,
        request: str | None = None,
        *,
        interpretation: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        if interpretation is None:
            if not request:
                raise ValueError("CapabilityAnalyzer.analyze requires request or interpretation")
            interpretation = self.request_interpreter.interpret(request, provider_name=provider_name, model=model, timeout=timeout)
        request_text = str(request or interpretation.get("request") or "")
        tools = self._available_tools()
        skills = self._available_skills()
        tool_ids = {t.get("id") for t in tools}
        skill_ids = {s.get("id") for s in skills}

        source_spaces = [str(s) for s in interpretation.get("source_spaces", []) if s]
        recommended_tools = self._normalize_tool_recommendations(interpretation.get("tools", []), tool_ids)
        recommended_skills = self._normalize_skill_recommendations(interpretation.get("skills", []), skill_ids)
        gaps = self._collect_gaps(interpretation, recommended_tools, recommended_skills, request_text)
        actions = self._recommended_actions(interpretation, source_spaces, recommended_tools, recommended_skills, gaps)
        priority = self._priority(gaps, actions)
        confidence = self._confidence(interpretation, gaps)

        return {
            "kind": "capability_analysis",
            "request": request_text,
            "intent": interpretation.get("intent", "unknown"),
            "summary": interpretation.get("summary", request_text[:160]),
            "source_spaces": source_spaces,
            "recommended_tools": recommended_tools,
            "recommended_skills": recommended_skills,
            "gaps": gaps,
            "has_gaps": bool(gaps),
            "gap_summary": {
                "tool": sum(1 for g in gaps if g["type"] == "tool"),
                "skill": sum(1 for g in gaps if g["type"] == "skill"),
                "knowledge": sum(1 for g in gaps if g["type"] == "knowledge"),
                "core": sum(1 for g in gaps if g["type"] == "core"),
            },
            "recommended_actions": actions,
            "priority": priority,
            "confidence": confidence,
            "available": {"tool_count": len(tools), "skill_count": len(skills)},
            "analysis_rule": "LLM recommends capability needs; Python validates availability and creates reviewable proposals only.",
            "safety": {
                "executes_tools": False,
                "generates_code": False,
                "activates_tools": False,
                "requires_user_approval_for_changes": True,
            },
            "interpreter": interpretation.get("interpreter", {}),
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

    def _normalize_tool_recommendations(self, items: Any, tool_ids: set[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            tool_id = str(item.get("id") or item.get("name") or "").strip()
            if not tool_id:
                continue
            out.append({
                "id": tool_id,
                "required": bool(item.get("required", False)),
                "available": tool_id in tool_ids,
                "reason": str(item.get("reason") or ""),
            })
        return out

    def _normalize_skill_recommendations(self, items: Any, skill_ids: set[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("id") or item.get("name") or "").strip()
            if not skill_id:
                continue
            out.append({
                "id": skill_id,
                "required": bool(item.get("required", False)),
                "available": skill_id in skill_ids,
                "reason": str(item.get("reason") or ""),
            })
        return out

    def _collect_gaps(
        self,
        interpretation: dict[str, Any],
        tools: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        request: str,
    ) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add_gap(kind: str, name: str, reason: str, source: str = "analyzer", severity: str = "medium") -> None:
            kind = kind if kind in GAP_TYPES else "knowledge"
            name = str(name or "unknown").strip() or "unknown"
            key = (kind, name.lower())
            if key in seen:
                return
            seen.add(key)
            gaps.append({"type": kind, "name": name, "reason": reason, "source": source, "severity": severity})

        for tool in tools:
            if tool.get("required") and not tool.get("available"):
                add_gap("tool", tool.get("id", "unknown"), tool.get("reason") or "Required tool is not available.", "tool_registry", "high")
        for skill in skills:
            if skill.get("required") and not skill.get("available"):
                add_gap("skill", skill.get("id", "unknown"), skill.get("reason") or "Required skill is not available.", "skill_registry", "high")
        for gap in interpretation.get("capability_gaps", []) or []:
            if isinstance(gap, dict):
                add_gap(str(gap.get("type") or "knowledge"), str(gap.get("name") or gap.get("capability") or "unknown"), str(gap.get("reason") or "Interpreter reported a capability gap."), "request_interpreter", "medium")

        # MVP 30.2 cleanup: no request-text keyword heuristics here.
        # Capability gaps may only come from structured interpreter/LLM output
        # or registry availability checks above.
        return gaps

    def _recommended_actions(
        self,
        interpretation: dict[str, Any],
        sources: list[str],
        tools: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        gaps: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        actions: list[dict[str, str]] = []

        def add(action: str, reason: str) -> None:
            if not any(a["action"] == action for a in actions):
                actions.append({"action": action, "reason": reason})

        if sources:
            add("context_lookup", "Recommended source spaces exist and should be collected by Python.")
        if any(t.get("available") and t.get("required") for t in tools):
            add("tool_use_review", "A required tool exists; Python must validate policy before execution.")
        if any(s.get("available") and s.get("required") for s in skills):
            add("skill_use_review", "A required skill exists; Python must validate policy before execution.")
        for gap in gaps:
            if gap["type"] == "tool":
                add("tool_factory_proposal", "Missing/requested tool must enter Tool Factory review/test/approval workflow.")
            elif gap["type"] == "skill":
                add("skill_proposal", "Missing skill must enter proposal and validation workflow.")
            elif gap["type"] == "knowledge":
                add("knowledge_update_proposal", "Missing knowledge should become a reviewable knowledge improvement proposal.")
            elif gap["type"] == "core":
                add("core_review_proposal", "Core change must become a reviewed architecture proposal, not an automatic modification.")
        if not actions:
            next_step = str(interpretation.get("recommended_next_step") or "answer")
            add("answer" if next_step == "answer" else next_step, "No missing capability requires proposal workflow.")
        return actions

    def _priority(self, gaps: list[dict[str, Any]], actions: list[dict[str, str]]) -> list[str]:
        order = ["core_review_proposal", "tool_factory_proposal", "skill_proposal", "knowledge_update_proposal", "tool_use_review", "skill_use_review", "context_lookup", "answer", "clarify"]
        action_names = [a["action"] for a in actions]
        return [a for a in order if a in action_names]

    def _confidence(self, interpretation: dict[str, Any], gaps: list[dict[str, Any]]) -> float:
        try:
            base = max(0.0, min(1.0, float(interpretation.get("confidence", 0.6))))
        except Exception:
            base = 0.6
        if gaps:
            return round(max(base, 0.7), 2)
        return round(base, 2)
