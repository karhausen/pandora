from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cognitive_planning_engine import CognitivePlanningEngine
from .python_orchestrator import ALLOWED_SOURCE_SPACES, SOURCE_POLICY
from .llm_config import LLMConfig
from .model_router import ModelRouter

SOURCE_ALIASES = {
    "obsidian": "obsidian_vault",
    "vault": "obsidian_vault",
    "notes": "obsidian_vault",
    "notizen": "obsidian_vault",
    "knowledge": "user_knowledge",
    "wissen": "user_knowledge",
    "memory": "long_term_memory",
    "conversation": "conversation_memory",
    "chat": "conversation_memory",
    "capabilities": "capability_graph",
    "capability": "capability_graph",
    "tools": "tool_registry",
    "tool": "tool_registry",
    "skills": "skill_registry",
    "skill": "skill_registry",
    "learning": "learning_engine",
}

SOURCE_DEFAULT_PRIORITY = {
    "obsidian_vault": 90,
    "user_knowledge": 82,
    "conversation_memory": 76,
    "long_term_memory": 70,
    "capability_graph": 64,
    "learning_engine": 58,
    "tool_registry": 52,
    "skill_registry": 50,
}

INTENT_SOURCE_BOOSTS = {
    "knowledge_lookup": {"obsidian_vault": 20, "user_knowledge": 14, "conversation_memory": 8, "long_term_memory": 8},
    "knowledge_summary": {"obsidian_vault": 14, "user_knowledge": 18, "long_term_memory": 8},
    "tool_request": {"tool_registry": 22, "capability_graph": 12, "learning_engine": 6},
    "tool_gap": {"tool_registry": 22, "capability_graph": 12, "learning_engine": 6},
    "core_improvement": {"learning_engine": 18, "capability_graph": 16, "user_knowledge": 8},
    "system_status": {"capability_graph": 14, "learning_engine": 10, "tool_registry": 6, "skill_registry": 6},
}

PLAN_MODE_SOURCE_BOOSTS = {
    "context_lookup": {"obsidian_vault": 12, "user_knowledge": 10, "conversation_memory": 8},
    "tool_proposal": {"tool_registry": 18, "capability_graph": 10},
    "knowledge_proposal": {"user_knowledge": 16, "obsidian_vault": 12},
    "core_proposal": {"learning_engine": 16, "capability_graph": 12},
}


@dataclass
class AdaptiveSourceSelector:
    """Selects source spaces from a cognitive plan without reading any source.

    The LLM/Request Interpreter may recommend semantic source spaces, but Python
    normalizes, validates and orders them. This keeps the flexible meaning
    understanding while preserving deterministic governance boundaries.
    """

    planning_engine: CognitivePlanningEngine | None = None
    llm_config: LLMConfig | None = None

    def __post_init__(self) -> None:
        self.planning_engine = self.planning_engine or CognitivePlanningEngine()
        self.llm_config = self.llm_config or LLMConfig()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "adaptive_source_selection_status",
            "ok": True,
            "mvp": "27.1",
            "role": "adaptive_source_space_selection_before_context_reading",
            "inputs": ["user_request", "cognitive_plan", "profile_policy"],
            "outputs": ["selected_sources", "blocked_sources", "source_trace"],
            "known_sources": sorted(ALLOWED_SOURCE_SPACES),
            "guarantee": "No file access, no tool execution, no code generation and no final answer.",
        }

    def select(
        self,
        request: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        max_sources: int = 5,
    ) -> dict[str, Any]:
        plan = self.planning_engine.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        profile = self._profile(provider_name=provider_name, model=model)
        candidates = self._collect_candidates(plan)
        ranked = self._rank_candidates(candidates, plan=plan, profile=profile)
        allowed = [c for c in ranked if c["allowed"]]
        selected = allowed[: max(1, int(max_sources))]
        blocked = [c for c in ranked if not c["allowed"]]
        return {
            "kind": "adaptive_source_selection",
            "request": request,
            "profile": profile,
            "plan_mode": plan.get("plan_mode"),
            "intent": plan.get("intent", "unknown"),
            "selection_status": "ready_for_context_builder" if selected else "no_allowed_sources_selected",
            "selected_sources": selected,
            "blocked_sources": blocked,
            "source_trace": {
                "plan_required_context": plan.get("required_context", []),
                "candidate_count": len(candidates),
                "ranked_count": len(ranked),
                "selected_count": len(selected),
                "policy": sorted(SOURCE_POLICY.get(profile, SOURCE_POLICY["local"])),
            },
            "safety": {
                "reads_files": False,
                "executes_tools": False,
                "generates_code": False,
                "writes_files": False,
                "python_validates_sources": True,
                "llm_recommends_only": True,
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
            route = ModelRouter(self.llm_config).route("planning", provider_name_override=provider_name, model_override=model)
            resolved = (route.provider_name or "").lower()
            if "company" in resolved:
                return "company"
            if "cloud" in resolved or "openai" in resolved:
                return "cloud"
        except Exception:
            pass
        return "local"

    def _collect_candidates(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        raw_sources: list[str] = []
        for source in plan.get("required_context", []) or []:
            raw_sources.append(str(source))
        trace = plan.get("trace", {}) if isinstance(plan.get("trace"), dict) else {}
        for key in ("interpreter", "central_decision"):
            value = trace.get(key, {}) if isinstance(trace.get(key), dict) else {}
            for source in value.get("source_spaces", []) or []:
                raw_sources.append(str(source))
        # Proposal modes should still inspect registries/architecture, even when
        # the interpreter did not explicitly name them.
        mode = str(plan.get("plan_mode") or "")
        if mode == "tool_proposal":
            raw_sources.extend(["tool_registry", "capability_graph"])
        elif mode == "knowledge_proposal":
            raw_sources.extend(["user_knowledge", "obsidian_vault"])
        elif mode == "core_proposal":
            raw_sources.extend(["learning_engine", "capability_graph"])
        elif not raw_sources and mode in {"context_lookup", "answer"}:
            raw_sources.extend(["conversation_memory"])

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_sources:
            normalized = self._normalize_source(raw)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append({"source": normalized, "raw": raw})
        return out

    def _normalize_source(self, raw: str) -> str | None:
        token = raw.strip().lower().replace("-", "_").replace(" ", "_")
        token = SOURCE_ALIASES.get(token, token)
        if token in ALLOWED_SOURCE_SPACES:
            return token
        return token if token else None

    def _rank_candidates(self, candidates: list[dict[str, Any]], *, plan: dict[str, Any], profile: str) -> list[dict[str, Any]]:
        allowed_policy = SOURCE_POLICY.get(profile, SOURCE_POLICY["local"])
        intent = str(plan.get("intent") or "unknown")
        mode = str(plan.get("plan_mode") or "answer")
        out: list[dict[str, Any]] = []
        for candidate in candidates:
            source = candidate["source"]
            known = source in ALLOWED_SOURCE_SPACES
            allowed = known and source in allowed_policy
            score = SOURCE_DEFAULT_PRIORITY.get(source, 10)
            score += INTENT_SOURCE_BOOSTS.get(intent, {}).get(source, 0)
            score += PLAN_MODE_SOURCE_BOOSTS.get(mode, {}).get(source, 0)
            if not known:
                score -= 80
            if not allowed:
                score -= 40
            reasons = []
            if source in PLAN_MODE_SOURCE_BOOSTS.get(mode, {}):
                reasons.append(f"fits_plan_mode:{mode}")
            if source in INTENT_SOURCE_BOOSTS.get(intent, {}):
                reasons.append(f"fits_intent:{intent}")
            if known:
                reasons.append("known_source_space")
            if allowed:
                reasons.append(f"allowed_for_{profile}")
            else:
                reasons.append("blocked_by_policy" if known else "unknown_source_space")
            out.append({
                "source": source,
                "raw": candidate.get("raw"),
                "score": score,
                "rank_reason": reasons,
                "known": known,
                "allowed": allowed,
                "profile": profile,
            })
        return sorted(out, key=lambda item: (-item["score"], item["source"]))
