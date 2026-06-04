from __future__ import annotations

from core.models import SecurityLevel, ToolDesign
from core.tool_review_agent import ToolReviewAgent
from core.tool_validator import ToolValidator


def _network_design() -> ToolDesign:
    return ToolDesign(
        capability="weather_lookup",
        tool_id="weather_lookup_tool",
        name="Weather Lookup Tool",
        description="Fetches current weather information for a location.",
        input_schema={"location": "string"},
        output_schema={"temperature": "float", "condition": "string"},
        security_level=SecurityLevel.LIMITED,
        requires_network=True,
        requires_filesystem=False,
        requires_shell=False,
    )


def test_policy_allows_urllib_for_limited_network_tool_with_timeout():
    code = '''
import json
import os
import urllib.parse
import urllib.request

TOOL_META = {"id": "weather_lookup_tool"}

def run(payload: dict) -> dict:
    location = payload.get("location", "")
    url = "https://api.example.invalid/weather?" + urllib.parse.urlencode({"q": location})
    with urllib.request.urlopen(url, timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
    return {"temperature": float(data.get("temperature", 0)), "condition": str(data.get("condition", "unknown"))}
'''
    review = ToolReviewAgent().review(code, design=_network_design())
    assert review["ok"] is True
    assert review["risk"] == "MEDIUM"
    assert review["policy"]["network_imports_allowed"] is True


def test_policy_rejects_requests_even_for_limited_network_tool():
    code = '''
import requests
TOOL_META = {"id": "weather_lookup_tool"}
def run(payload: dict) -> dict:
    return requests.get("https://example.invalid").json()
'''
    review = ToolValidator().static_review(code, design=_network_design())
    assert review["ok"] is False
    assert "Forbidden import: requests" in review["issues"]


def test_policy_rejects_urllib_for_safe_tool():
    design = _network_design()
    design.requires_network = False
    design.security_level = SecurityLevel.SAFE
    code = '''
import urllib.request
TOOL_META = {"id": "bad_safe_tool"}
def run(payload: dict) -> dict:
    return {}
'''
    review = ToolReviewAgent().review(code, design=design)
    assert review["ok"] is False
    assert "Forbidden import: urllib.request" in review["issues"]


def test_policy_rejects_urlopen_without_timeout():
    code = '''
import urllib.request
TOOL_META = {"id": "weather_lookup_tool"}
def run(payload: dict) -> dict:
    urllib.request.urlopen("https://example.invalid")
    return {}
'''
    review = ToolReviewAgent().review(code, design=_network_design())
    assert review["ok"] is False
    assert "Network call must set timeout keyword" in review["issues"]
