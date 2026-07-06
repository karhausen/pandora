from core.chat_service import ChatService


def test_chat_service_uses_llm_led_route_registry_components():
    service = ChatService()
    assert hasattr(service, "route_registry")
    assert hasattr(service, "route_planner")
    assert {r.id for r in service.route_registry.available_specs()} == {
        "direct_answer",
        "vault_search",
        "memory_search",
        "clarify_user",
    }
