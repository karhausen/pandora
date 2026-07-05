from core.capability_snapshot import CapabilitySnapshotBuilder
from core.capability_model import CapabilityRecord
from pathlib import Path


def test_snapshot_exposes_unified_capabilities():
    snapshot = CapabilitySnapshotBuilder().build()
    dumped = snapshot.model_dump()
    assert "capabilities" in dumped
    assert isinstance(dumped["capabilities"], list)
    kinds = {cap["kind"] for cap in dumped["capabilities"]}
    assert "knowledge" in kinds
    assert "memory" in kinds
    assert "workflow" in kinds
    assert "tool" in kinds or dumped["tools"] == []
    assert any(cap["id"] == "knowledge:obsidian_vault" for cap in dumped["capabilities"])
    assert any(cap["id"] == "workflow:tool_factory" for cap in dumped["capabilities"])


def test_capability_record_is_neutral_and_llm_readable():
    record = CapabilityRecord(
        id="knowledge:test",
        name="Test Knowledge",
        kind="knowledge",
        description="Test source",
        permissions=["read_knowledge"],
    ).model_dump()
    assert record["kind"] == "knowledge"
    assert record["implementation_ref"] is None
    assert "permissions" in record


def test_orchestrator_prompt_uses_unified_capabilities_not_categories_only():
    source = Path("core/capability_orchestrator.py").read_text(encoding="utf-8")
    assert "snapshot.capabilities" in source
    assert "CapabilityRecord" in source
    assert "needed_capabilities" in source
