from fastapi.testclient import TestClient
from core.api import app


def test_night_review_web_route_exists():
    client = TestClient(app)
    response = client.get("/night-review")
    assert response.status_code == 200
    assert "Night Review" in response.text


def test_night_review_assets_exist():
    client = TestClient(app)
    assert client.get("/web/night-review.js").status_code == 200
    assert client.get("/web/night-review.css").status_code == 200


def test_web_routes_diagnostics_includes_night_review():
    client = TestClient(app)
    data = client.get("/api/system/web-routes").json()
    assert data["version"] == "24.8.1-night-review-web-route-fix"
    paths = {item["path"] for item in data["routes"]}
    assert "/night-review" in paths
