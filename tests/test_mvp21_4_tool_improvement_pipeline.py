from __future__ import annotations

import json
from pathlib import Path

from core.tool_improvement_pipeline import ToolImprovementPipeline


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sample_registry() -> dict:
    return {
        "good_tool": {
            "id": "good_tool",
            "name": "Good Tool",
            "description": "stable",
            "status": "ACTIVE",
            "module": "tools.good",
            "function": "run",
            "input_schema": {"text": "str"},
            "output_schema": {"text": "str"},
        },
        "bad_tool": {
            "id": "bad_tool",
            "name": "Bad Tool",
            "description": "fails often",
            "status": "ACTIVE",
            "module": "tools.bad",
            "function": "run",
            "input_schema": {"text": "str"},
            "output_schema": {"result": "str"},
        },
    }


def sample_stats() -> dict:
    return {
        "good_tool": {"executions": 10, "successes": 10, "failures": 0},
        "bad_tool": {
            "executions": 10,
            "successes": 3,
            "failures": 7,
            "last_error": "schema mismatch",
            "last_used": "2026-06-08T00:00:00+00:00",
            "total_execution_time": 4.2,
        },
    }


def test_status_is_observe_only(tmp_path: Path):
    pipeline = ToolImprovementPipeline(
        registry_file=tmp_path / "registry.json",
        stats_file=tmp_path / "stats.json",
        output_dir=tmp_path / "proposals",
    )
    status = pipeline.status()
    assert status["observe_only"] is True
    assert "modify tool source code" in status["blocked_actions"]


def test_detects_weak_tool_from_stats(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    stats_file = tmp_path / "stats.json"
    write_json(registry_file, sample_registry())
    write_json(stats_file, sample_stats())
    pipeline = ToolImprovementPipeline(registry_file=registry_file, stats_file=stats_file, output_dir=tmp_path / "proposals")

    result = pipeline.run_once(force=True, dry_run=True)

    assert result["status"] == "planned"
    assert result["activated"] is False
    assert result["observe_only"] is True
    assert result["proposal"]["tool"]["tool_id"] == "bad_tool"
    assert result["proposal"]["review_required"] is True


def test_writes_reviewable_proposal_without_modifying_registry(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    stats_file = tmp_path / "stats.json"
    registry = sample_registry()
    write_json(registry_file, registry)
    write_json(stats_file, sample_stats())
    pipeline = ToolImprovementPipeline(registry_file=registry_file, stats_file=stats_file, output_dir=tmp_path / "proposals")

    result = pipeline.run_once(force=True, dry_run=False)

    assert result["status"] == "completed"
    assert Path(result["written_to"]).exists()
    proposal = json.loads(Path(result["written_to"]).read_text(encoding="utf-8"))
    assert proposal["kind"] == "tool_improvement_proposal"
    assert proposal["tool"]["tool_id"] == "bad_tool"
    assert proposal["activated"] is False
    assert json.loads(registry_file.read_text(encoding="utf-8")) == registry


def test_no_candidate_when_tools_are_healthy(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    stats_file = tmp_path / "stats.json"
    write_json(registry_file, {"good_tool": sample_registry()["good_tool"]})
    write_json(stats_file, {"good_tool": sample_stats()["good_tool"]})
    pipeline = ToolImprovementPipeline(registry_file=registry_file, stats_file=stats_file, output_dir=tmp_path / "proposals")

    result = pipeline.run_once(force=True)

    assert result["status"] == "no_candidate"
    assert not (tmp_path / "proposals").exists()


def test_blocks_without_registry(tmp_path: Path):
    pipeline = ToolImprovementPipeline(registry_file=tmp_path / "missing.json", stats_file=tmp_path / "stats.json", output_dir=tmp_path / "out")
    result = pipeline.run_once(force=True)
    assert result["status"] == "skipped"
    assert "tool registry file not found" in result["decision"]["reasons"]
