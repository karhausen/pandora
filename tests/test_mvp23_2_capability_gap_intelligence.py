from fastapi.testclient import TestClient

from core.api import app
from core.capability_graph import CapabilityGraphService
from core.capability_gap_intelligence import CapabilityGapIntelligenceService


def test_capability_gap_intelligence_scores_missing_tool_and_skill(tmp_path):
    graph_dir = tmp_path / "graph"
    service = CapabilityGraphService(graph_dir=graph_dir)
    service._write({
        "kind": "capability_graph",
        "updated_at": "2026-06-10T00:00:00Z",
        "nodes": [
            {"id": "cap:funktechnik", "label": "funktechnik", "type": "capability", "source": "test", "metadata": {}},
            {"id": "knowledge:public:funk.md", "label": "Funktechnik", "type": "knowledge", "source": "test", "metadata": {}},
        ],
        "edges": [
            {"source": "cap:funktechnik", "target": "knowledge:public:funk.md", "relation": "has_knowledge", "weight": 1, "metadata": {}},
        ],
        "summary": {},
    })
    report = CapabilityGapIntelligenceService(service).analyze(limit=10)
    assert report["kind"] == "capability_gap_intelligence_report"
    assert report["findings"]
    finding = report["findings"][0]
    assert finding["capability_id"] == "cap:funktechnik"
    assert "knowledge exists but no tool is linked" in finding["reasons"]
    assert "knowledge exists but no skill is linked" in finding["reasons"]


def test_capability_intelligence_api_and_gui_assets():
    client = TestClient(app)
    page = client.get('/capability-explorer')
    assert page.status_code == 200
    assert 'Capability Gap Intelligence' in page.text

    js = client.get('/web/capability-explorer.js')
    assert js.status_code == 200
    assert 'loadIntelligence' in js.text

    report = client.get('/api/capabilities/intelligence?limit=5')
    assert report.status_code == 200
    assert report.json()["kind"] == "capability_gap_intelligence_report"

    rebuilt = client.post('/api/capabilities/intelligence/rebuild?limit=5')
    assert rebuilt.status_code == 200
    assert rebuilt.json()["safety"]["read_only"] is True
