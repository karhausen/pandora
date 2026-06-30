from core.cognitive_integration_regression import CognitiveIntegrationRegressionService


class FakeDecisionEngine:
    def __init__(self, decision_type="tool_gap", approval=True):
        self.decision_type = decision_type
        self.approval = approval

    def decide(self, request, **kwargs):
        mode = {
            "tool_gap": "tool_proposal",
            "core_gap": "core_proposal",
            "knowledge_gap": "knowledge_proposal",
            "context_answer": "context_lookup",
        }.get(self.decision_type, "answer")
        return {
            "kind": "central_decision",
            "request": request,
            "decision_type": self.decision_type,
            "execution_mode": mode,
            "status": "requires_user_decision" if self.approval else "ready_for_safe_processing",
            "summary": "fake decision",
            "requires_user_approval": self.approval,
            "approval_prompt": "Wir brauchen ein Tool 'x'. Soll ich den Tool-Vorschlag ausarbeiten?" if self.approval else None,
            "next_controlled_step": "await_user_approval_to_prepare_tool_factory_proposal" if self.approval else "continue_to_context_builder_and_prompt_builder",
            "source_spaces": ["obsidian_vault"] if self.decision_type == "context_answer" else [],
            "safety": {"executes_tools": False, "generates_code": False, "writes_files": False, "activates_tools": False, "changes_core": False},
        }


class FakeApprovalWorkflow:
    def preview(self, request, **kwargs):
        return {
            "kind": "approval_interaction_preview",
            "interaction_state": "awaiting_user_decision" if kwargs.get("user_decision") is None else "approved_for_proposal_preparation",
            "short_user_message": "Soll ich den Vorschlag ausarbeiten?",
            "safety": {"executes_tools": False, "generates_code": False, "writes_files": False, "activates_tools": False, "changes_core": False},
        }


class FakeReviewLoop:
    def preview(self, request, **kwargs):
        return {
            "kind": "proposal_review_loop_preview",
            "review_state": "approved_for_next_controlled_step",
            "short_user_message": "Review freigegeben.",
            "safety": {"executes_tools": False, "generates_code": False, "writes_files": False, "activates_tools": False, "changes_core": False},
        }


class FakeExecutionGate:
    def preview(self, request, **kwargs):
        return {
            "kind": "proposal_execution_gate_preview",
            "gate_state": "waiting_for_final_execution_approval",
            "short_user_message": "Warte auf finale Freigabe.",
            "safety": {"activates_tools": False, "writes_knowledge": False, "changes_core": False, "creates_release": False},
        }


class FakeContextPipeline:
    def preview(self, request, **kwargs):
        return {
            "kind": "cognitive_context_pipeline_preview",
            "pipeline_status": "context_ready",
            "context": {"items": [{"source": "obsidian_vault"}]},
            "safety": {"executes_tools": False},
        }


def service(decision_type="tool_gap", approval=True):
    return CognitiveIntegrationRegressionService(
        decision_engine=FakeDecisionEngine(decision_type, approval),
        approval_workflow=FakeApprovalWorkflow(),
        review_loop=FakeReviewLoop(),
        execution_gate=FakeExecutionGate(),
        context_pipeline=FakeContextPipeline(),
    )


def test_status_lists_integrated_components():
    status = service().status()
    assert status["ok"] is True
    assert status["mvp"] == "26.5"
    assert "central_decision_engine" in status["integrates"]
    assert "obsidian_last_note_context" in status["regression_scenarios"]


def test_preview_creates_single_trace_without_execution():
    preview = service().preview("Ich brauche ein Tool fuer Aktienkurse", user_decision="ja", review_decision="passt")
    assert preview["kind"] == "cognitive_integration_preview"
    assert preview["decision"]["decision_type"] == "tool_gap"
    assert [step["step"] for step in preview["trace"]][:3] == ["central_decision", "approval_interaction", "proposal_review_loop"]
    assert preview["safety"]["dangerous_action_detected"] is False


def test_context_answer_runs_context_trace_without_approval():
    preview = service("context_answer", approval=False).preview("Was war meine letzte Notiz?")
    assert preview["decision"]["requires_user_approval"] is False
    assert any(step["step"] == "context_pipeline" for step in preview["trace"])
    assert preview["context_pipeline"]["pipeline_status"] == "context_ready"


def test_regression_report_detects_expected_context_flow():
    svc = service("context_answer", approval=False)
    svc.scenarios = [{
        "id": "obsidian_last_note_context",
        "request": "Was war meine letzte Notiz?",
        "expected_decision_types": ["context_answer"],
        "expected_next_steps": ["continue_to_context_builder_and_prompt_builder"],
        "expected_sources_any": ["obsidian_vault"],
        "must_not_require_approval": True,
        "purpose": "guard",
    }]
    report = svc.run_regression(timeout=0.01)
    assert report["ok"] is True
    assert report["passed"] == 1
