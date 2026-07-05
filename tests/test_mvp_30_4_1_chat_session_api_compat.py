from core.chat_service import ChatService


def test_chat_service_session_api_compatibility():
    service = ChatService()
    session = service.create_session("API compatibility")
    assert session["session_id"]

    sessions = service.list_sessions()
    assert isinstance(sessions, list)
    assert any(item["session_id"] == session["session_id"] for item in sessions)

    loaded = service.get_session(session["session_id"])
    assert loaded["session_id"] == session["session_id"]

    deleted = service.delete_session(session["session_id"])
    assert deleted["deleted"] is True
