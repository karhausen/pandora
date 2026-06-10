from pathlib import Path

from core.capability_graph import CapabilityGraphService, capability_id
from core.user_knowledge_base import UserKnowledgeBaseService


def test_capability_graph_rebuild_from_knowledge(tmp_path):
    kb_root = tmp_path / "user_knowledge"
    service = UserKnowledgeBaseService(root_dir=kb_root)
    service.ensure_structure()
    note = kb_root / "public" / "funktechnik" / "kalibrierung.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "---\n"
        "title: Funkgerät Kalibrierung\n"
        "tags:\n"
        "  - funktechnik\n"
        "  - kalibrierung\n"
        "visibility: public\n"
        "cloud_allowed: true\n"
        "priority: high\n"
        "---\n\n"
        "# Kalibrierung\n\nMessablauf für Funkgeräte.",
        encoding="utf-8",
    )

    graph_dir = tmp_path / "data" / "capability_graph"
    graph = CapabilityGraphService(graph_dir=graph_dir)
    # Monkeypatch the class-level knowledge root by injecting through service implementation path.
    # For this MVP test we temporarily patch the default root on the imported service class instance.
    from core import capability_graph as module

    original = module.UserKnowledgeBaseService
    module.UserKnowledgeBaseService = lambda: service
    try:
        payload = graph.rebuild(write=True)
    finally:
        module.UserKnowledgeBaseService = original

    assert payload["summary"]["node_count"] >= 2
    assert graph.graph_file.exists()
    ids = {node["id"] for node in payload["nodes"]}
    assert capability_id("kalibrierung") in ids
    assert any(edge["relation"] == "has_knowledge" for edge in payload["edges"])


def test_capability_graph_show_missing(tmp_path):
    graph = CapabilityGraphService(graph_dir=tmp_path / "graph")
    result = graph.show_capability("does-not-exist")
    assert result["found"] is False
