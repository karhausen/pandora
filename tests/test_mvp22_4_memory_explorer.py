from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from core.api import app
from core.memory_explorer import MemoryExplorerService


def test_memory_explorer_dashboard_reads_allowed_dirs(tmp_path: Path):
    memory = tmp_path / "memory"
    proposals = tmp_path / "proposals"
    memory.mkdir()
    proposals.mkdir()
    (memory / "core_events.jsonl").write_text('{"kind":"test"}\n', encoding="utf-8")
    (proposals / "review.json").write_text(json.dumps({"title": "Review"}), encoding="utf-8")

    service = MemoryExplorerService(memory_dir=memory, proposals_dir=proposals)
    dashboard = service.dashboard()

    assert dashboard["read_only"] is True
    assert dashboard["area_count"] == 2
    assert dashboard["total_files"] == 2


def test_memory_explorer_search(tmp_path: Path):
    memory = tmp_path / "memory"
    proposals = tmp_path / "proposals"
    memory.mkdir()
    proposals.mkdir()
    (memory / "agent_journal.jsonl").write_text('{"task":"calculate voltage"}\n', encoding="utf-8")

    service = MemoryExplorerService(memory_dir=memory, proposals_dir=proposals)
    result = service.search(query="voltage")

    assert result["count"] == 1
    assert result["results"][0]["area"] == "memory"


def test_memory_explorer_blocks_path_escape(tmp_path: Path):
    memory = tmp_path / "memory"
    proposals = tmp_path / "proposals"
    memory.mkdir()
    proposals.mkdir()
    service = MemoryExplorerService(memory_dir=memory, proposals_dir=proposals)

    try:
        service.show_file("memory", "../secret.json")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape was not blocked")


def test_memory_explorer_api_and_page():
    client = TestClient(app)

    page = client.get("/memory-explorer")
    assert page.status_code == 200
    assert "Memory Explorer" in page.text

    dashboard = client.get("/api/gui/memory/dashboard")
    assert dashboard.status_code == 200
    payload = dashboard.json()
    assert payload["kind"] == "memory_explorer_dashboard"
    assert payload["read_only"] is True


def test_user_gui_links_memory_explorer():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "/memory-explorer" in page.text
    assert "Memory Explorer" in page.text
