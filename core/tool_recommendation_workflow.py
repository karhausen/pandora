from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .python_orchestrator import PythonOrchestrator


DEFAULT_SECURITY_RULES = [
    "LLM output is a proposal only and must never be activated automatically.",
    "Generated code must stay inside the approved tool interface.",
    "No filesystem, network, shell or credential access unless explicitly declared and approved.",
    "All generated tools require tests, governance review and user approval before registry activation.",
]


@dataclass
class ToolRecommendationWorkflow:
    """Prepares reviewable Tool Factory inputs from diagnosed tool gaps.

    This workflow does not call an LLM for code generation, does not write tool
    files, does not activate registry entries and does not execute anything. It
    converts a validated Python orchestration plan into a Tool Factory proposal
    brief that can later be reviewed, tested and approved.
    """

    python_orchestrator: PythonOrchestrator | None = None

    def __post_init__(self) -> None:
        self.python_orchestrator = self.python_orchestrator or PythonOrchestrator()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "tool_recommendation_workflow_status",
            "ok": True,
            "role": "tool_gap_to_tool_factory_proposal_brief",
            "pipeline_position": "after_python_orchestrator_before_tool_factory_code_generation",
            "guarantee": "No code generation, no tool execution, no file writes, no registry activation.",
            "outputs": ["tool_factory_briefs", "review_steps", "test_requirements", "approval_requirements"],
        }

    def prepare(
        self,
        request: str | None = None,
        *,
        orchestration_plan: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        if orchestration_plan is None:
            if not request:
                raise ValueError("ToolRecommendationWorkflow.prepare requires request or orchestration_plan")
            orchestration_plan = self.python_orchestrator.plan(request, provider_name=provider_name, model=model, timeout=timeout)
        request_text = str(request or orchestration_plan.get("request") or "")
        tool_gaps = self._tool_gaps(orchestration_plan)
        briefs = [self._brief_from_gap(gap, request_text) for gap in tool_gaps]
        return {
            "kind": "tool_recommendation_workflow_preview",
            "request": request_text,
            "plan_status": orchestration_plan.get("plan_status"),
            "tool_gap_count": len(tool_gaps),
            "tool_factory_briefs": briefs,
            "recommended_next_step": "review_tool_factory_briefs" if briefs else "no_tool_gap_detected",
            "requires_user_approval": bool(briefs),
            "safety": {
                "generates_code": False,
                "executes_tools": False,
                "writes_files": False,
                "activates_tools": False,
                "hands_to_tool_factory_only_after_review": True,
            },
            "orchestration_plan": orchestration_plan,
        }

    def _tool_gaps(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for gap in plan.get("gap_plan", []) or []:
            if not isinstance(gap, dict) or gap.get("type") != "tool":
                continue
            name = str(gap.get("name") or "requested_tool").strip() or "requested_tool"
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            gaps.append(gap)
        return gaps

    def _brief_from_gap(self, gap: dict[str, Any], request: str) -> dict[str, Any]:
        tool_id = self._normalize_tool_id(str(gap.get("name") or "requested_tool"))
        purpose = str(gap.get("reason") or f"Provide missing capability for request: {request}")
        input_schema, output_schema = self._schemas_for(tool_id, request)
        return {
            "status": "draft_requires_review",
            "tool_id": tool_id,
            "name": tool_id.replace("_", " ").title(),
            "purpose": purpose,
            "source_request": request,
            "interface_contract": {
                "entrypoint": "run(payload: dict) -> dict",
                "input_schema": input_schema,
                "output_schema": output_schema,
                "error_contract": {"ok": False, "error": "string", "tool_id": tool_id},
            },
            "llm_generation_brief": {
                "task": "Generate Python code for this tool only inside the declared interface.",
                "must_include": ["TOOL_META", "run(payload: dict) -> dict", "input validation", "deterministic error handling"],
                "must_not_include": ["automatic activation", "credential logging", "unapproved shell execution", "unapproved network calls"],
            },
            "test_requirements": self._test_requirements(tool_id, input_schema, output_schema),
            "review_workflow": [
                "tool_factory_proposal_created",
                "code_generation_review",
                "unit_tests",
                "security_governance_check",
                "user_approval",
                "registry_activation",
                "post_activation_learning_review",
            ],
            "security_rules": DEFAULT_SECURITY_RULES,
            "requires_user_approval": True,
            "severity": str(gap.get("severity") or "medium"),
        }

    def _normalize_tool_id(self, raw: str) -> str:
        value = raw.strip().lower().replace(" ", "_").replace("-", "_")
        value = "".join(ch for ch in value if ch.isalnum() or ch == "_").strip("_")
        if not value or value == "requested_tool":
            return "requested_tool"
        return value

    def _schemas_for(self, tool_id: str, request: str) -> tuple[dict[str, str], dict[str, str]]:
        text = f"{tool_id} {request}".lower()
        if any(word in text for word in ["stock", "aktien", "börse", "boerse", "kurs"]):
            return (
                {"ticker": "string", "start_date": "string optional YYYY-MM-DD", "end_date": "string optional YYYY-MM-DD"},
                {"ok": "boolean", "ticker": "string", "time_series": "list", "metadata": "dict", "source": "string"},
            )
        if any(word in text for word in ["csv", "excel", "xlsx", "datei"]):
            return (
                {"path": "string", "options": "dict optional"},
                {"ok": "boolean", "rows": "number", "columns": "list", "summary": "dict"},
            )
        return (
            {"text": "string optional", "input": "string optional", "options": "dict optional"},
            {"ok": "boolean", "result": "any", "metadata": "dict"},
        )

    def _test_requirements(self, tool_id: str, input_schema: dict[str, str], output_schema: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {"name": f"test_{tool_id}_valid_input", "purpose": "Tool returns ok=True and declared output keys for valid payload."},
            {"name": f"test_{tool_id}_invalid_input", "purpose": "Tool returns ok=False with deterministic error for invalid payload."},
            {"name": f"test_{tool_id}_interface_contract", "purpose": "Tool exposes TOOL_META and run(payload: dict) -> dict."},
        ]
