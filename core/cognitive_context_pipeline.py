from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cognitive_context_builder import CognitiveContextBuilder
from .request_interpreter import RequestInterpreter
from .capability_analyzer import CapabilityAnalyzer
from .python_orchestrator import PythonOrchestrator
from .llm_config import LLMConfig


PIPELINE_STEPS = [
    "request_interpretation",
    "capability_analysis",
    "python_orchestration",
    "context_collection",
    "context_ranking",
    "duplicate_removal",
    "context_budget",
    "prompt_context_ready",
]


@dataclass
class CognitiveContextPipeline:
    """Traceable cognitive context pipeline preview.

    This service connects the Request Interpreter, Capability Analyzer,
    Python Orchestrator and Cognitive Context Builder into one auditable flow.
    It is deliberately a preview/trace layer: it does not execute tools, does
    not generate code, does not activate registries and does not change core.
    """

    llm_config: LLMConfig | None = None
    request_interpreter: RequestInterpreter | None = None
    capability_analyzer: CapabilityAnalyzer | None = None
    python_orchestrator: PythonOrchestrator | None = None
    context_builder: CognitiveContextBuilder | None = None

    def __post_init__(self) -> None:
        self.llm_config = self.llm_config or LLMConfig()
        self.request_interpreter = self.request_interpreter or RequestInterpreter(llm_config=self.llm_config)
        self.capability_analyzer = self.capability_analyzer or CapabilityAnalyzer(request_interpreter=self.request_interpreter)
        self.python_orchestrator = self.python_orchestrator or PythonOrchestrator(capability_analyzer=self.capability_analyzer, llm_config=self.llm_config)
        self.context_builder = self.context_builder or CognitiveContextBuilder(
            llm_config=self.llm_config,
            request_interpreter=self.request_interpreter,
            capability_analyzer=self.capability_analyzer,
            python_orchestrator=self.python_orchestrator,
        )

    def status(self) -> dict[str, Any]:
        return {
            "kind": "cognitive_context_pipeline_status",
            "ok": True,
            "role": "auditable_preview_pipeline",
            "guarantee": "No tool execution, no code generation, no registry activation, no core modification.",
            "steps": PIPELINE_STEPS,
            "components": {
                "request_interpreter": self.request_interpreter.status(),
                "capability_analyzer": self.capability_analyzer.status(),
                "python_orchestrator": self.python_orchestrator.status(),
                "context_builder": self.context_builder.status(),
            },
        }

    def preview(
        self,
        request: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        limit: int | None = 5,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        interpretation = self.request_interpreter.interpret(request, provider_name=provider_name, model=model, timeout=timeout)
        analysis = self.capability_analyzer.analyze(request, interpretation=interpretation)
        orchestration = self.python_orchestrator.plan(request, analysis=analysis, provider_name=provider_name, model=model, timeout=timeout)
        context = self.context_builder.build_for_chat(request, provider_name=provider_name, model=model, limit=limit)
        trace = self._trace(interpretation, analysis, orchestration, context)
        return {
            "kind": "cognitive_context_pipeline_preview",
            "request": request,
            "pipeline_status": self._pipeline_status(orchestration, context),
            "steps": trace,
            "request_interpretation": interpretation,
            "capability_analysis": analysis,
            "orchestration_plan": orchestration,
            "context": {
                "target": context.get("target"),
                "route_target": context.get("route_target"),
                "source_count": context.get("source_count", 0),
                "context_chars": context.get("context_chars", 0),
                "sources": context.get("sources", []),
                "context_ranking": context.get("context_ranking", {}),
                "policy": context.get("policy", {}),
                "context_text": context.get("context_text", ""),
            },
            "safety": {
                "executes_tools": False,
                "generates_code": False,
                "activates_tools": False,
                "changes_core": False,
                "llm_reads_files_directly": False,
                "python_validates_before_action": True,
            },
        }

    def _pipeline_status(self, orchestration: dict[str, Any], context: dict[str, Any]) -> str:
        if orchestration.get("plan_status") == "blocked_by_policy":
            return "blocked_by_policy"
        if orchestration.get("requires_user_approval"):
            return "needs_user_approval"
        if context.get("source_count", 0) > 0 and context.get("context_chars", 0) > 0:
            return "context_ready"
        return "no_context_found"

    def _trace(self, interpretation: dict[str, Any], analysis: dict[str, Any], orchestration: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
        ranking = context.get("context_ranking", {}) or {}
        return [
            {
                "step": "request_interpretation",
                "ok": True,
                "intent": interpretation.get("intent"),
                "source_spaces": interpretation.get("source_spaces", []),
                "tool_recommendations": len(interpretation.get("tools", []) or []),
                "confidence": interpretation.get("confidence"),
            },
            {
                "step": "capability_analysis",
                "ok": True,
                "gap_count": len(analysis.get("gaps", []) or []),
                "recommended_actions": analysis.get("recommended_actions", []) or analysis.get("priority", []),
            },
            {
                "step": "python_orchestration",
                "ok": orchestration.get("plan_status") != "blocked_by_policy",
                "plan_status": orchestration.get("plan_status"),
                "requires_user_approval": orchestration.get("requires_user_approval"),
                "blocked_count": orchestration.get("blocked_count", 0),
            },
            {
                "step": "context_collection",
                "ok": True,
                "source_count": context.get("source_count", 0),
                "blocked_local_only_count": context.get("diagnostics", {}).get("blocked_local_only_count", 0),
                "blocked_obsidian_count": context.get("diagnostics", {}).get("blocked_obsidian_count", 0),
            },
            {
                "step": "context_ranking",
                "ok": True,
                "selected_count": ranking.get("selected_count", context.get("source_count", 0)),
                "candidate_count": ranking.get("candidate_count"),
            },
            {
                "step": "duplicate_removal",
                "ok": True,
                "duplicates_removed": ranking.get("duplicates_removed", 0),
            },
            {
                "step": "context_budget",
                "ok": True,
                "context_chars": context.get("context_chars", 0),
                "budget_chars": ranking.get("budget_chars"),
            },
            {
                "step": "prompt_context_ready",
                "ok": context.get("context_chars", 0) > 0,
                "context_embedded": context.get("context_chars", 0) > 0,
            },
        ]
