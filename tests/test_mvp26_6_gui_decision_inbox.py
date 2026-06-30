from core.gui_decision_inbox import GuiDecisionInbox


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
            "approval_prompt": "Wir brauchen ein Tool 'stock_history_lookup'. Soll ich den Tool-Vorschlag ausarbeiten?" if self.approval else None,
            "next_controlled_step": "await_user_approval_to_prepare_tool_factory_proposal" if self.approval else "continue_to_context_builder_and_prompt_builder",
            "confidence": 0.91,
            "priority": ["tool"],
            "source_spaces": ["obsidian_vault"] if self.decision_type == "context_answer" else [],
            "gap_types": ["tool"] if self.decision_type == "tool_gap" else [],
            "review_packages": {"tool": {"tool_factory_briefs": [{"tool_id": "stock_history_lookup"}]}} if self.decision_type == "tool_gap" else {},
            "safety": {"executes_tools": False, "generates_code": False, "writes_files": False, "activates_tools": False, "changes_core": False},
            "orchestration_plan": {},
        }


def test_status_is_safe_gui_adapter():
    status = GuiDecisionInbox(FakeDecisionEngine()).status()
    assert status["ok"] is True
    assert status["mvp"] == "26.6"
    assert "prepare_proposal" in status["actions"]


def test_tool_gap_card_asks_for_simple_proposal_approval():
    preview = GuiDecisionInbox(FakeDecisionEngine("tool_gap", True)).preview("Ich brauche ein Aktien Tool")
    card = preview["cards"][0]
    assert card["title"] == "Tool wird benötigt"
    assert card["requires_user_approval"] is True
    assert [a["id"] for a in card["actions"]] == ["prepare_proposal", "defer", "reject"]
    assert preview["action_result"]["state"] == "awaiting_user_action"
    assert preview["safety"]["activates_tools"] is False


def test_prepare_proposal_action_creates_handoff_without_execution():
    preview = GuiDecisionInbox(FakeDecisionEngine("tool_gap", True)).preview("Ich brauche ein Tool", user_action="ja")
    assert preview["selected_action"] == "prepare_proposal"
    assert preview["action_result"]["state"] == "proposal_preparation_approved"
    assert preview["action_result"]["next_step"] == "proposal_review_loop_with_tool_factory_payload"
    assert preview["safety"]["generates_code"] is False


def test_context_answer_can_continue_without_proposal():
    preview = GuiDecisionInbox(FakeDecisionEngine("context_answer", False)).preview("Was war meine letzte Notiz?")
    card = preview["cards"][0]
    assert card["title"] == "Kontext-Antwort"
    assert card["requires_user_approval"] is False
    assert card["actions"][0]["id"] == "continue"
