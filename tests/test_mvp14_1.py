from fastapi.testclient import TestClient
from core.api import app

client = TestClient(app)

def test_dashboard_full_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "Agent Run" in r.text
    assert "Learning" in r.text
    assert "Governance" in r.text

def test_dashboard_js_has_agent_run():
    r = client.get("/web/app.js")
    assert r.status_code == 200
    assert "runAgent" in r.text
    assert "provider_name" in r.text

def test_dashboard_css():
    r = client.get("/web/style.css")
    assert r.status_code == 200
    assert ".grid" in r.text
