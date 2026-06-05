from __future__ import annotations

from pathlib import Path

from core.models import SecurityLevel, ToolSpec
from core.tool_quality_gate import ToolQualityGate
from core.tool_proposal_manager import ToolProposalManager


def _proposal_dir(tmp_path: Path, tool_id: str, code: str) -> Path:
    proposal_dir = tmp_path / "proposal"
    tool_dir = proposal_dir / "generated_tools"
    tool_dir.mkdir(parents=True)
    (tool_dir / "__init__.py").write_text("", encoding="utf-8")
    (tool_dir / f"{tool_id}.py").write_text(code, encoding="utf-8")
    return proposal_dir


def test_quality_gate_rejects_output_schema_mismatch(tmp_path: Path):
    spec = ToolSpec(
        id="word_counter_bad",
        name="Word Counter Bad",
        description="Counts words",
        capability="word_count",
        input_schema={"text": "string"},
        output_schema={"count": "integer"},
        security_level=SecurityLevel.SAFE,
    )
    code = '''TOOL_META = {
    "id": "word_counter_bad",
    "name": "Word Counter Bad",
    "description": "Counts words",
    "version": "0.1.0",
    "input_schema": {"text": "string"},
    "output_schema": {"count": "integer"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.word_counter_bad",
    "function": "run",
}

def run(payload: dict) -> dict:
    return {"text": payload.get("text", "")}
'''
    result = ToolQualityGate().validate(_proposal_dir(tmp_path, spec.id, code), spec.id, spec)

    assert result["ok"] is False
    assert any("missing schema key: count" in issue for issue in result["issues"])


def test_quality_gate_accepts_matching_word_counter(tmp_path: Path):
    spec = ToolSpec(
        id="word_counter_ok",
        name="Word Counter OK",
        description="Counts words",
        capability="word_count",
        input_schema={"text": "string"},
        output_schema={"count": "integer"},
        security_level=SecurityLevel.SAFE,
    )
    code = '''TOOL_META = {
    "id": "word_counter_ok",
    "name": "Word Counter OK",
    "description": "Counts words",
    "version": "0.1.0",
    "input_schema": {"text": "string"},
    "output_schema": {"count": "integer"},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.word_counter_ok",
    "function": "run",
}

def run(payload: dict) -> dict:
    return {"count": len(str(payload.get("text", "")).split())}
'''
    result = ToolQualityGate().validate(_proposal_dir(tmp_path, spec.id, code), spec.id, spec)

    assert result["ok"] is True
    assert result["checks"]["cases"][0]["output"] == {"count": 3}


def test_proposal_generation_includes_semantic_validation():
    proposal = ToolProposalManager().propose_for_capability("word_count", provider_name="mock")

    assert "semantic" in proposal["validation"]
    assert proposal["validation"]["semantic"]["ok"] is True
    assert proposal["status"] == "VALIDATED"
