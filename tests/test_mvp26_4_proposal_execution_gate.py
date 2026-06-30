from core.proposal_execution_gate import ProposalExecutionGate


def test_execution_gate_status_is_safe():
    status = ProposalExecutionGate().status()
    assert status["ok"] is True
    assert "no direct activation" in status["guarantee"].lower()
    assert "execution_handoff" in status["outputs"]


def test_gate_waits_for_final_execution_approval_after_review():
    payload = {"purpose": "historische Aktienkurse", "python_code": "def run(payload): return {}"}
    result = ProposalExecutionGate().preview("Baue ein Tool für historische Aktienkurse", proposal_payload=payload, review_decision="passt")
    assert result["gate_state"] == "waiting_for_final_execution_approval"
    assert result["execution_handoff"]["allowed"] is False
    assert result["safety"]["activates_tools"] is False


def test_tool_gate_blocks_without_tests_and_audit():
    payload = {"purpose": "historische Aktienkurse", "python_code": "def run(payload): return {}"}
    result = ProposalExecutionGate().preview(
        "Baue ein Tool für historische Aktienkurse",
        proposal_payload=payload,
        review_decision="passt",
        execution_decision="aktivieren",
    )
    assert result["gate_state"] == "blocked_until_required_checks_pass"
    assert result["check_results"]["ok"] is False
    assert result["execution_handoff"]["allowed"] is False


def test_tool_gate_allows_only_controlled_handoff_when_checks_pass():
    payload = {"purpose": "historische Aktienkurse", "python_code": "def run(payload): return {}"}
    result = ProposalExecutionGate().preview(
        "Baue ein Tool für historische Aktienkurse",
        proposal_payload=payload,
        review_decision="passt",
        execution_decision="aktivieren",
        test_report={"ok": True},
        audit_report={"ok": True},
    )
    assert result["gate_state"] == "ready_for_controlled_handoff"
    assert result["execution_handoff"]["allowed"] is True
    assert result["execution_handoff"]["target_workflow"] == "tool_activation_or_registry_gate"
    assert result["safety"]["activates_tools"] is False


def test_gate_blocks_when_review_needs_work():
    payload = {"purpose": "Core verbessern"}
    result = ProposalExecutionGate().preview(
        "Pandora Core verbessern",
        proposal_payload=payload,
        review_decision="nachbessern",
        execution_decision="aktivieren",
        test_report={"ok": True},
        audit_report={"ok": True},
    )
    assert result["gate_state"] == "blocked_until_review_approved"
    assert result["execution_handoff"]["allowed"] is False
