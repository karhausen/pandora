from __future__ import annotations

from pathlib import Path

from core.capability_graph import CapabilityGraphService
from core.obsidian_vault import ObsidianConfig, ObsidianVaultService


def test_capability_graph_collects_obsidian_tags_and_wikilinks(tmp_path: Path, monkeypatch):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "Funktechnik.md").write_text(
        "# Funktechnik\n\nSpektrumanalyse und Kalibrierung. #funktechnik #messtechnik\n\nSiehe [[Kalibrierung]].\n",
        encoding="utf-8",
    )
    config = ObsidianConfig(
        enabled=True,
        vault_path=vault_dir,
        inbox_dir="Pandora_Inbox",
        mode="read_write_inbox_only",
        cloud_allowed=False,
    )

    class TestVault(ObsidianVaultService):
        def __init__(self):
            super().__init__(root_dir=tmp_path, config=config)

    import core.capability_graph as capability_graph

    monkeypatch.setattr(capability_graph, "ObsidianVaultService", TestVault)
    service = CapabilityGraphService(graph_dir=tmp_path / "capability_graph")
    graph = service.rebuild(write=True)
    node_types = {node["type"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}
    labels = {node["label"] for node in graph["nodes"] if node["type"] == "capability"}

    assert "obsidian_note" in node_types
    assert "has_obsidian_note" in relations
    assert "funktechnik" in labels
    assert "Kalibrierung".lower() in {label.lower() for label in labels}
    assert service.status()["node_count"] >= 2


def test_capability_graph_ignores_disabled_obsidian(tmp_path: Path):
    service = CapabilityGraphService(graph_dir=tmp_path / "capability_graph")
    graph = service.rebuild(write=False)
    assert graph["kind"] == "capability_graph"
    assert all(node.get("type") != "obsidian_note" for node in graph["nodes"])
