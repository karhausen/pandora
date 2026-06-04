from __future__ import annotations

from core.cloud_tool_code_generator import CloudToolCodeGenerator
from core.models import ToolDesign
from core.tool_design_agent import ToolDesignAgent


def test_policy_aware_test_adjustments_add_imports_and_env_setup():
    generator = CloudToolCodeGenerator()
    design = ToolDesign(
        capability="weather_lookup",
        tool_id="weather_lookup_tool",
        name="Weather Lookup Tool",
        description="Fetches weather.",
        input_schema={"location": "string"},
        output_schema={"temperature": "float"},
        security_level="LIMITED",
        requires_network=True,
    )
    code = """
import os
import urllib.request

TOOL_META = {}

def run(payload: dict) -> dict:
    api_key = os.getenv('WEATHER_API_KEY')
    with urllib.request.urlopen('https://example.test', timeout=5) as resp:
        return {'temperature': 20.0}
"""
    test_code = """
from generated_tools.weather_lookup_tool import run

def test_basic():
    def mock_urlopen(url, timeout):
        class MockResponse:
            def read(self):
                return json.dumps({'temperature': 20.0}).encode('utf-8')
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockResponse()
    monkeypatch.setattr(urllib.request, 'urlopen', mock_urlopen)
    assert run({'location': 'New York'})['temperature'] == 20.0
"""

    fixed = generator._policy_aware_test_adjustments(test_code, code, design)

    assert "import json" in fixed
    assert "import urllib.request" in fixed
    assert "def test_basic(monkeypatch):" in fixed
    assert "monkeypatch.setenv('WEATHER_API_KEY', 'test-value')" in fixed


def test_tool_design_agent_removes_forbidden_dependencies():
    agent = ToolDesignAgent()
    design = agent._validate_design(
        {
            "capability": "weather_lookup",
            "tool_id": "weather_lookup_tool",
            "name": "Weather Lookup Tool",
            "description": "Fetch weather.",
            "input_schema": {"location": "string"},
            "output_schema": {"temperature": "float"},
            "security_level": "LIMITED",
            "requires_network": True,
            "dependencies": ["requests", "json"],
            "test_cases": [],
            "implementation_notes": [],
            "risk_notes": [],
            "confidence": 0.9,
        },
        capability="weather_lookup",
        source="test",
    )

    assert "requests" not in design.dependencies
    assert "json" in design.dependencies
    assert any("Removed forbidden external dependencies" in note for note in design.risk_notes)
