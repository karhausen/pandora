
from fastapi.testclient import TestClient
from core.api import app

client = TestClient(app)

def test_index():
    r = client.get("/")
    assert r.status_code == 200

def test_js():
    r = client.get("/web/app.js")
    assert r.status_code == 200
