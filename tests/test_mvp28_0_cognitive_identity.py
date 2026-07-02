from core.cognitive_identity import CognitiveIdentityService


def test_cognitive_identity_status_is_read_only():
    status = CognitiveIdentityService().status()
    assert status["ok"] is True
    assert status["mvp"] == "28.0"
    assert "Read-only" in status["guarantee"]


def test_identity_card_contains_positive_and_negative_identity():
    card = CognitiveIdentityService().identity_card()
    assert card["name"] == "Pandora"
    assert card["core_identity"]["is"]
    assert card["core_identity"]["is_not"]
    assert "not allowed to silently change its own core" in card["core_identity"]["is_not"]


def test_boundaries_require_approval_before_changes():
    boundaries = CognitiveIdentityService().capability_boundaries()
    assert "modifying core files" in boundaries["must_ask_or_stop_before"]
    assert "claiming success when tests or audits were not run" in boundaries["must_ask_or_stop_before"]


def test_self_model_without_request_has_no_request_assessment():
    model = CognitiveIdentityService().self_model()
    assert model["mvp"] == "28.0"
    assert model["request_self_assessment"] is None
    assert model["safe_operating_statement"]["execution_allowed_by_this_service"] is False
