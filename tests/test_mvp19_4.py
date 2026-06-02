from __future__ import annotations

from fastapi.testclient import TestClient

from core.api import app
from core.llm_config import LLMConfig
from core.llm_router import LLMRouter
from core.model_router import ModelRouter
from core.models import LLMTaskType

client = TestClient(app)


def test_model_router_routes_daily_work_local_and_generation_cloud():
    router = ModelRouter()

    chat = router.route("chat")
    selection = router.route("tool_selection")
    generation = router.route("tool_generation")
    review = router.route("core_review")

    assert chat.provider_name == "local_fast"
    assert selection.provider_name == "local_fast"
    assert generation.provider_name == "openai"
    assert review.provider_name == "openai"
    assert generation.resolved_from == "model_routes"


def test_model_router_honors_lmstudio_override_alias():
    route = ModelRouter().route("tool_selection", provider_name_override="lmstudio")

    assert route.provider_name == "local_fast"
    assert route.requested_provider_name == "lmstudio"
    assert route.resolved_from == "override"


def test_legacy_llm_router_delegates_to_model_router():
    route = LLMRouter().route(LLMTaskType.TOOL_GENERATION)

    assert route.provider_name == "openai"
    assert route.provider.value == "openai"
    assert "expert" in route.reason.lower() or "generation" in route.reason.lower()


def test_model_router_api_routes_are_available():
    response = client.get("/model-router/routes")
    assert response.status_code == 200
    data = response.json()
    assert data["routes"]["chat"]["provider_name"] == "local_fast"
    assert data["routes"]["tool_generation"]["provider_name"] == "openai"

    response = client.get("/model-router/route/tool_selection?provider_name=lmstudio")
    assert response.status_code == 200
    assert response.json()["provider_name"] == "local_fast"
