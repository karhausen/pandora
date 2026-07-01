from core.adaptive_source_selection import AdaptiveSourceSelector


class FakePlanningEngine:
    def __init__(self, plan):
        self._plan = plan

    def plan(self, request, **kwargs):
        return self._plan


def test_status_is_safe_and_plan_only():
    status = AdaptiveSourceSelector(FakePlanningEngine({})).status()
    assert status["mvp"] == "27.1"
    assert status["ok"] is True
    assert "No file access" in status["guarantee"]


def test_selects_obsidian_for_knowledge_lookup():
    plan = {
        "plan_mode": "context_lookup",
        "intent": "knowledge_lookup",
        "required_context": ["obsidian_vault", "conversation_memory", "obsidian"],
        "trace": {},
    }
    result = AdaptiveSourceSelector(FakePlanningEngine(plan)).select("Was war meine letzte Notiz?", max_sources=3)
    sources = [item["source"] for item in result["selected_sources"]]
    assert sources[0] == "obsidian_vault"
    assert "conversation_memory" in sources
    assert result["safety"]["reads_files"] is False


def test_tool_proposal_adds_registry_sources():
    plan = {
        "plan_mode": "tool_proposal",
        "intent": "tool_request",
        "required_context": [],
        "trace": {},
    }
    result = AdaptiveSourceSelector(FakePlanningEngine(plan)).select("Ich brauche ein Aktien Tool")
    sources = [item["source"] for item in result["selected_sources"]]
    assert "tool_registry" in sources
    assert "capability_graph" in sources


def test_cloud_profile_blocks_obsidian_vault():
    plan = {
        "plan_mode": "context_lookup",
        "intent": "knowledge_lookup",
        "required_context": ["obsidian_vault", "user_knowledge", "conversation_memory"],
        "trace": {},
    }
    result = AdaptiveSourceSelector(FakePlanningEngine(plan)).select("Was war meine letzte Notiz?", provider_name="openai")
    blocked_sources = [item["source"] for item in result["blocked_sources"]]
    assert "obsidian_vault" in blocked_sources
    selected = [item["source"] for item in result["selected_sources"]]
    assert "conversation_memory" in selected
