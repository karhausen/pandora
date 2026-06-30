from core.central_decision_engine import CentralDecisionEngine


def test_central_decision_status_is_safe():
    status = CentralDecisionEngine().status()
    assert status["ok"] is True
    assert "No execution" in status["guarantee"]
    assert "decision_object" in status["outputs"]


def test_central_decision_tool_gap_asks_simple_approval():
    engine = CentralDecisionEngine()
    decision = engine.decide("Baue ein Tool für historische Aktienkurse", include_review_packages=True)
    assert decision["kind"] == "central_decision"
    assert decision["decision_type"] in {"tool_gap", "mixed_capability_review"}
    assert decision["requires_user_approval"] is True
    assert "Soll ich" in decision["approval_prompt"]
    assert decision["safety"]["generates_code"] is False
    assert decision["safety"]["activates_tools"] is False
    assert "tool" in decision["gap_types"]


def test_central_decision_core_gap_routes_to_core_proposal():
    engine = CentralDecisionEngine()
    decision = engine.decide("Pandora Core verbessern: zentrale Decision Engine", include_review_packages=True)
    assert decision["requires_user_approval"] is True
    assert "core" in decision["gap_types"]
    assert decision["execution_mode"] in {"core_proposal", "mixed_review"}
    assert decision["safety"]["changes_core"] is False


def test_central_decision_context_lookup_does_not_need_approval_for_safe_sources():
    engine = CentralDecisionEngine()
    decision = engine.decide("Was war meine letzte Notiz?", include_review_packages=False)
    assert decision["decision_type"] in {"context_answer", "direct_answer"}
    assert decision["execution_mode"] in {"context_lookup", "answer"}
    assert decision["safety"]["reads_files"] is False
