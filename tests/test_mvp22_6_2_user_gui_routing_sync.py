from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app


def test_user_status_exposes_active_chat_route_from_router():
    client = TestClient(app)
    response = client.get("/user/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_selection_mode"] == "central_routing"
    assert payload["active_chat_route"]["purpose"] == "chat"
    assert payload["active_chat_route"]["provider_name"]
    assert payload["routing_editor_url"] == "/llm-profiles"


def test_user_gui_no_longer_contains_local_provider_select():
    html = Path("web/index.html").read_text(encoding="utf-8")
    js = Path("web/user.js").read_text(encoding="utf-8")
    assert 'id="providerSelect"' not in html
    assert 'id="modelInput"' not in html
    assert "activeChatProvider" in html
    assert "renderActiveChatRoute" in js
    assert "localStorage.getItem(\"pandora_provider\")" not in js
    assert "provider_name: currentProvider" not in js


def test_user_gui_links_provider_changes_to_llm_profile_center():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "Chat-Route" in html
    assert 'href="/llm-profiles"' in html
    assert "Routing ändern" in html


def test_user_run_request_does_not_default_to_mock_provider():
    from core.api import UserRunRequest

    req = UserRunRequest(task="Hallo")
    assert req.provider_name is None


def test_chat_path_defaults_use_central_routing_not_mock_override():
    from pathlib import Path

    assert 'provider_name: str | None = None' in Path('core/chat_service.py').read_text(encoding='utf-8')
    assert 'provider_name: str | None = None' in Path('core/llm_chat_responder.py').read_text(encoding='utf-8')
    assert 'provider_name: str | None = None' in Path('core/coordinator_agent.py').read_text(encoding='utf-8')
