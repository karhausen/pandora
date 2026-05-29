import asyncio

from core.reality_check import RealityCheck
from core.stability_reporter import StabilityReporter
from fastapi.testclient import TestClient
from core.api import app

client = TestClient(app)


def test_reality_check_single_iteration():
    result = asyncio.run(RealityCheck().run(iterations=1, delay=0))
    assert result.iterations == 1
    assert result.passed + result.failed == 1
    assert result.recommendations


def test_stability_reporter():
    report = StabilityReporter().report()
    assert "snapshots" in report
    assert "memory" in report


def test_reality_api_and_dashboard():
    response = client.get("/reality-check/report")
    assert response.status_code == 200
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Reality Check" in dashboard.text
