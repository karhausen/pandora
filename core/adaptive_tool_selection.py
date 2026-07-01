from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from .cognitive_planning_engine import CognitivePlanningEngine
from .tool_registry import ToolRegistry
from .models import SecurityLevel, ToolStatus
from .llm_config import LLMConfig
from .model_router import ModelRouter

TOOL_ALIASES = {
    "calculate": "calculator",
    "calc": "calculator",
    "rechnung": "calculator",
    "rechnen": "calculator",
    "math": "calculator",
    "uppercase": "uppercase",
    "gross": "uppercase",
    "groß": "uppercase",
    "echo": "echo",
}

TOOL_SECURITY_POLICY = {
    "local": {"SAFE", "LIMITED"},
    "company": {"SAFE", "LIMITED"},
    "cloud": {"SAFE"},
}

PLAN_MODE_TOOL_BOOSTS = {
    "answer": 4,
    "context_lookup": 0,
    "tool_proposal": 18,
    "knowledge_proposal": 0,
    "core_proposal": 0,
}

INTENT_TOOL_BOOSTS = {
    "calculation": {"calculator": 35},
    "math": {"calculator": 35},
    "tool_request": {},
    "tool_gap": {},
    "code_generation": {},
}

KEYWORD_TOOL_HINTS = {
    "calculator": ["rechnen", "berechne", "calculate", "rechnung", "summe", "plus", "minus", "*", "/"],
    "uppercase": ["uppercase", "gross", "groß", "versalien", "majuskeln"],
    "echo": ["echo", "wiederhole"],
}

@dataclass
class AdaptiveToolSelector:
    """Recommends tools from the registry without executing them.

    The LLM/Cognitive Planner may describe the needed capability. Python then
    normalizes, checks the registry, scores available tools and detects gaps.
    This avoids brittle keyword-only selection while keeping execution and
    security decisions deterministic.
    """

    planning_engine: CognitivePlanningEngine | None = None
    registry: ToolRegistry | None = None
    llm_config: LLMConfig | None = None

    def __post_init__(self) -> None:
        self.planning_engine = self.planning_engine or CognitivePlanningEngine()
        self.registry = self.registry or ToolRegistry()
        self.llm_config = self.llm_config or LLMConfig()

    def status(self) -> dict[str, Any]:
        tools = self._tool_catalog()
        return {
            "kind": "adaptive_tool_selection_status",
            "ok": True,
            "mvp": "27.2",
            "role": "adaptive_tool_selection_before_execution",
            "inputs": ["user_request", "cognitive_plan", "tool_registry", "profile_policy"],
            "outputs": ["selected_tools", "blocked_tools", "tool_gaps", "tool_trace"],
            "registered_tools": [tool["id"] for tool in tools],
            "guarantee": "No tool execution, no code generation, no registry write and no final answer.",
        }

    def select(
        self,
        request: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_tools: int = 3,
    ) -> dict[str, Any]:
        plan = self.planning_engine.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        profile = self._profile(provider_name=provider_name, model=model)
        catalog = self._tool_catalog()
        requested = self._collect_requested_tools(plan, request)
        ranked = self._rank_tools(catalog, requested=requested, request=request, plan=plan, profile=profile)
        allowed = [item for item in ranked if item["allowed"]]
        selected = [item for item in allowed if item["score"] > 0][: max(0, int(max_tools))]
        blocked = [item for item in ranked if not item["allowed"] and item["score"] > 0]
        gaps = self._detect_gaps(requested=requested, selected=selected, plan=plan, request=request)
        return {
            "kind": "adaptive_tool_selection",
            "request": request,
            "profile": profile,
            "plan_mode": plan.get("plan_mode"),
            "intent": plan.get("intent", "unknown"),
            "selection_status": "tool_recommendation_ready" if selected else ("tool_gap_detected" if gaps else "no_tool_needed"),
            "selected_tools": selected,
            "blocked_tools": blocked,
            "tool_gaps": gaps,
            "tool_trace": {
                "requested_tools": requested,
                "registered_count": len(catalog),
                "ranked_count": len(ranked),
                "selected_count": len(selected),
                "gap_count": len(gaps),
                "policy_allowed_security_levels": sorted(TOOL_SECURITY_POLICY.get(profile, TOOL_SECURITY_POLICY["local"])),
            },
            "safety": {
                "executes_tools": False,
                "generates_code": False,
                "writes_registry": False,
                "python_validates_tools": True,
                "llm_recommends_only": True,
                "requires_user_approval_for_gaps": bool(gaps),
            },
            "cognitive_plan": plan,
        }

    def _profile(self, *, provider_name: str | None, model: str | None) -> str:
        raw_provider = (provider_name or "").lower()
        if "company" in raw_provider:
            return "company"
        if "cloud" in raw_provider or "openai" in raw_provider:
            return "cloud"
        try:
            route = ModelRouter(self.llm_config).route("tool_selection", provider_name_override=provider_name, model_override=model)
            resolved = (route.provider_name or "").lower()
            if "company" in resolved:
                return "company"
            if "cloud" in resolved or "openai" in resolved:
                return "cloud"
        except Exception:
            pass
        return "local"

    def _tool_catalog(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for meta in self.registry.list():
            data = meta.model_dump(mode="json")
            data["security_level"] = str(data.get("security_level", "SAFE")).split(".")[-1]
            data["status"] = str(data.get("status", "ACTIVE")).split(".")[-1]
            out.append(data)
        return sorted(out, key=lambda item: item["id"])

    def _collect_requested_tools(self, plan: dict[str, Any], request: str) -> list[dict[str, Any]]:
        raw: list[str] = []
        for key in ("tools", "required_tools", "recommended_tools"):
            for item in plan.get(key, []) or []:
                raw.append(str(item))
        trace = plan.get("trace", {}) if isinstance(plan.get("trace"), dict) else {}
        for value in trace.values():
            if isinstance(value, dict):
                for key in ("tools", "recommended_tools", "required_tools"):
                    for item in value.get(key, []) or []:
                        raw.append(str(item))
        text = request.lower()
        for tool_id, hints in KEYWORD_TOOL_HINTS.items():
            if any(hint in text for hint in hints):
                raw.append(tool_id)
        if str(plan.get("plan_mode")) == "tool_proposal":
            gap_name = plan.get("missing_capability") or plan.get("capability") or self._infer_missing_capability(request)
            raw.append(str(gap_name))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            normalized = self._normalize_tool(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append({"tool": normalized, "raw": item})
        return out

    def _normalize_tool(self, raw: str) -> str | None:
        token = raw.strip().lower().replace("-", "_").replace(" ", "_")
        token = TOOL_ALIASES.get(token, token)
        token = re.sub(r"[^a-z0-9_]+", "", token)
        return token or None

    def _rank_tools(self, catalog: list[dict[str, Any]], *, requested: list[dict[str, Any]], request: str, plan: dict[str, Any], profile: str) -> list[dict[str, Any]]:
        requested_ids = {item["tool"] for item in requested}
        allowed_levels = TOOL_SECURITY_POLICY.get(profile, TOOL_SECURITY_POLICY["local"])
        intent = str(plan.get("intent") or "unknown")
        mode = str(plan.get("plan_mode") or "answer")
        text = request.lower()
        out: list[dict[str, Any]] = []
        for tool in catalog:
            tool_id = str(tool.get("id"))
            status = str(tool.get("status", "ACTIVE"))
            security_level = str(tool.get("security_level", "SAFE"))
            active = status in {"ACTIVE", "VALIDATED"}
            allowed = active and security_level in allowed_levels
            score = 0
            reasons: list[str] = []
            if tool_id in requested_ids:
                score += 70; reasons.append("requested_by_cognitive_plan_or_hint")
            desc = f"{tool.get('name','')} {tool.get('description','')} {' '.join(tool.get('aliases') or [])}".lower()
            for word in self._content_words(text):
                if word in desc:
                    score += 4; reasons.append(f"matches_request_word:{word}")
            score += INTENT_TOOL_BOOSTS.get(intent, {}).get(tool_id, 0)
            if tool_id in INTENT_TOOL_BOOSTS.get(intent, {}):
                reasons.append(f"fits_intent:{intent}")
            score += PLAN_MODE_TOOL_BOOSTS.get(mode, 0) if tool_id in requested_ids else 0
            if mode == "tool_proposal" and tool_id in requested_ids:
                reasons.append("fits_tool_proposal_mode")
            if active:
                reasons.append("registered_tool_active_or_validated")
            else:
                score -= 50; reasons.append("tool_not_active")
            if allowed:
                reasons.append(f"allowed_for_{profile}")
            else:
                score -= 40; reasons.append("blocked_by_policy_or_status")
            out.append({
                "tool": tool_id,
                "name": tool.get("name"),
                "description": tool.get("description"),
                "score": score,
                "security_level": security_level,
                "status": status,
                "allowed": allowed,
                "profile": profile,
                "rank_reason": reasons,
                "input_schema": tool.get("input_schema", {}),
                "output_schema": tool.get("output_schema", {}),
            })
        return sorted(out, key=lambda item: (-item["score"], item["tool"]))

    def _detect_gaps(self, *, requested: list[dict[str, Any]], selected: list[dict[str, Any]], plan: dict[str, Any], request: str) -> list[dict[str, Any]]:
        selected_ids = {item["tool"] for item in selected}
        registered_ids = {meta.id for meta in self.registry.list()}
        gaps: list[dict[str, Any]] = []
        for item in requested:
            tool_id = item["tool"]
            if tool_id not in registered_ids and tool_id not in selected_ids:
                gaps.append(self._gap(tool_id, item.get("raw"), plan))
        if str(plan.get("plan_mode")) == "tool_proposal" and not gaps and not selected:
            gaps.append(self._gap(self._infer_missing_capability(request), None, plan))
        # Deduplicate by suggested id.
        unique: dict[str, dict[str, Any]] = {}
        for gap in gaps:
            unique[gap["suggested_tool_id"]] = gap
        return list(unique.values())

    def _gap(self, tool_id: str, raw: str | None, plan: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_tool(tool_id) or "missing_tool"
        return {
            "kind": "tool_gap",
            "suggested_tool_id": normalized,
            "raw_request": raw,
            "proposal_action": "ask_user_before_tool_factory",
            "requires_user_approval": True,
            "recommended_next_question": f"Wir brauchen ein Tool '{normalized}'. Soll ich einen Vorschlag bauen?",
            "interface_hint": {
                "input_schema": plan.get("suggested_input_schema", {}),
                "output_schema": plan.get("suggested_output_schema", {}),
            },
        }

    def _infer_missing_capability(self, request: str) -> str:
        text = request.lower()
        if "aktie" in text or "stock" in text or "börse" in text or "kurs" in text:
            return "stock_history_lookup"
        if "kalender" in text or "calendar" in text:
            return "calendar_lookup"
        return "requested_tool_capability"

    def _content_words(self, text: str) -> list[str]:
        stop = {"ich", "ein", "eine", "der", "die", "das", "und", "oder", "mit", "für", "brauche", "tool", "bitte", "mir", "mein"}
        return [w for w in re.findall(r"[a-zäöüß0-9_]{3,}", text.lower()) if w not in stop][:12]
