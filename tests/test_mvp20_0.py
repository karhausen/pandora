from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from core.config import GENERATED_TOOLS_DIR, TOOL_PROPOSALS_DIR
from core.tool_activation_manager import ToolActivationManager
from core.tool_proposal_manager import ToolProposalManager
from core.tool_registry import ToolRegistry


def _validated_fixture(proposal_id: str = "tool_lifecycle_test") -> dict:
    proposal_dir = TOOL_PROPOSALS_DIR / proposal_id
    tool_dir = proposal_dir / "generated_tools"
    test_dir = proposal_dir / "tests"
    tool_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "__init__.py").write_text("", encoding="utf-8")
    code_file = tool_dir / "unit_echo_install.py"
    code_file.write_text('''TOOL_META = {
    "id": "unit_echo_install",
    "name": "Unit Echo Install",
    "description": "Lifecycle test tool.",
    "version": "0.1.0",
    "input_schema": {"text": "str"},
    "output_schema": {"text": "str"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.unit_echo_install",
    "function": "run",
}

def run(payload: dict) -> dict:
    return {"text": payload.get("text", "")}
''', encoding="utf-8")
    test_file = test_dir / "test_unit_echo_install.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    proposal = {
        "id": proposal_id,
        "status": "VALIDATED",
        "capability": "unit_echo_install",
        "spec": {
            "id": "unit_echo_install",
            "name": "Unit Echo Install",
            "description": "Lifecycle test tool.",
            "capability": "unit_echo_install",
            "input_schema": {"text": "str"},
            "output_schema": {"text": "str"},
            "security_level": "SAFE",
        },
        "created_at": datetime.now(UTC).isoformat(),
        "proposal_dir": str(proposal_dir),
        "code_file": str(code_file),
        "test_file": str(test_file),
        "validation": {"static": {"ok": True}, "tests": {"success": True}},
        "design": None,
        "risk": "LOW",
    }
    (proposal_dir / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    return proposal


def _cleanup_installed() -> None:
    registry = ToolRegistry()
    registry.tools.pop("unit_echo_install", None)
    registry.save()
    installed_file = GENERATED_TOOLS_DIR / "unit_echo_install.py"
    if installed_file.exists():
        installed_file.unlink()


def test_proposal_lifecycle_approve_reject_guards():
    proposal = _validated_fixture("tool_lifecycle_approve_test")
    manager = ToolProposalManager()

    shown = manager.show(proposal["id"])["proposal"]
    assert shown["status"] == "VALIDATED"

    approved = manager.approve(proposal["id"], note="approved for lifecycle test")
    assert approved["success"] is True
    assert approved["status"] == "APPROVED"

    rejected = manager.reject(proposal["id"], reason="should still be rejectable before install")
    assert rejected["success"] is True
    assert rejected["status"] == "REJECTED"

    again = manager.approve(proposal["id"])
    assert again["success"] is False


class _FakeToolExecutor:
    def __init__(self, registry):
        self.registry = registry

    async def run_tool(self, tool_id: str, payload: dict, timeout: float = 5.0, task: str | None = None):
        return SimpleNamespace(success=True, error=None)


def test_approved_proposal_can_be_installed_and_registered(monkeypatch):
    from core import tool_activation_manager as activation_module

    _cleanup_installed()
    proposal = _validated_fixture("tool_lifecycle_install_test")
    manager = ToolProposalManager()

    monkeypatch.setattr(activation_module, "ToolExecutor", _FakeToolExecutor)

    not_installed = asyncio.run(ToolActivationManager().activate(proposal["id"], test_payload={"text": "hello"}))
    assert not_installed.activated is False
    assert "APPROVED" in (not_installed.error or "")

    approved = manager.approve(proposal["id"])
    assert approved["success"] is True

    installed = asyncio.run(ToolActivationManager().activate(proposal["id"], test_payload={"text": "hello"}))
    assert installed.activated is True
    assert installed.tool_id == "unit_echo_install"
    assert installed.registered is True
    assert installed.tested is True

    final = manager.show(proposal["id"])["proposal"]
    assert final["status"] == "INSTALLED"

    registry = ToolRegistry()
    registry.discover()
    assert registry.get("unit_echo_install") is not None
    _cleanup_installed()


def test_install_normalizes_cloud_design_style_tool_meta(monkeypatch):
    from core import tool_activation_manager as activation_module

    proposal_id = "tool_lifecycle_design_meta_test"
    proposal_dir = TOOL_PROPOSALS_DIR / proposal_id
    tool_dir = proposal_dir / "generated_tools"
    test_dir = proposal_dir / "tests"
    tool_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "__init__.py").write_text("", encoding="utf-8")

    code_file = tool_dir / "word_count_tool.py"
    code_file.write_text('''TOOL_META = {
    "tool_id": "word_count_tool",
    "name": "Word Count Tool",
    "description": "Counts words.",
    "input_schema": {"text": "string"},
    "output_schema": {"word_count": "integer"},
}

def run(payload: dict) -> dict:
    text = payload.get("text", "")
    return {"word_count": len([part for part in text.split() if part])}
''', encoding="utf-8")
    test_file = test_dir / "test_word_count_tool.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    proposal = {
        "id": proposal_id,
        "status": "VALIDATED",
        "capability": "word_count",
        "spec": {
            "id": "word_count_tool",
            "name": "Word Count Tool",
            "description": "Counts words.",
            "capability": "word_count",
            "input_schema": {"text": "string"},
            "output_schema": {"word_count": "integer"},
            "security_level": "SAFE",
        },
        "created_at": datetime.now(UTC).isoformat(),
        "proposal_dir": str(proposal_dir),
        "code_file": str(code_file),
        "test_file": str(test_file),
        "validation": {"static": {"ok": True}, "tests": {"success": True}},
        "design": {
            "capability": "word_count",
            "tool_id": "word_count_tool",
            "name": "Word Count Tool",
            "description": "Counts words.",
            "input_schema": {"text": "string"},
            "output_schema": {"word_count": "integer"},
            "security_level": "SAFE",
        },
        "risk": "LOW",
    }
    (proposal_dir / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    registry = ToolRegistry()
    registry.tools.pop("word_count_tool", None)
    registry.save()
    installed_file = GENERATED_TOOLS_DIR / "word_count_tool.py"
    if installed_file.exists():
        installed_file.unlink()

    monkeypatch.setattr(activation_module, "ToolExecutor", _FakeToolExecutor)
    approved = ToolProposalManager().approve(proposal_id)
    assert approved["success"] is True

    installed = asyncio.run(ToolActivationManager().activate(proposal_id, test_payload={"text": "eins zwei drei"}))
    assert installed.activated is True
    assert installed.tool_id == "word_count_tool"

    registry = ToolRegistry()
    registry.discover()
    meta = registry.get("word_count_tool")
    assert meta is not None
    assert meta.module == "generated_tools.word_count_tool"
    assert meta.id == "word_count_tool"

    registry.tools.pop("word_count_tool", None)
    registry.save()
    if installed_file.exists():
        installed_file.unlink()
