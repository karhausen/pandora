from core.operations_cockpit import OperationsCockpitService


def test_operations_cockpit_dashboard_shape():
    data = OperationsCockpitService().dashboard(limit=10)
    assert data["kind"] == "operations_cockpit_dashboard"
    assert "headline" in data
    assert "sections" in data
    assert data["safety"]["auto_execute_actions"] is False


def test_operations_cockpit_quick_links_include_core_pages():
    data = OperationsCockpitService().dashboard(limit=10)
    hrefs = {item["href"] for item in data["quick_links"]}
    assert "/action-inbox" in hrefs
    assert "/workflow-dashboard" in hrefs
    assert "/review-scheduler" in hrefs
