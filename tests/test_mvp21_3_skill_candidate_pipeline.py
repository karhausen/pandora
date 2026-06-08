from __future__ import annotations

import json
from pathlib import Path

from core.skill_candidate_pipeline import SkillCandidatePipeline
from core.skill_proposal_manager import SkillProposalManager
from core.task_journal import TaskJournal


def _append_tool(journal: TaskJournal, tool_id: str) -> None:
    journal.append({
        "success": True,
        "action": {"type": "tool", "tool_id": tool_id},
        "result": {"route": "tool_execution"},
    })


def test_skill_candidate_status_is_observe_only(tmp_path: Path):
    journal = TaskJournal(tmp_path / "memory" / "agent_journal.jsonl")
    manager = SkillProposalManager()
    manager.root = tmp_path / "skill_proposals"
    manager.root.mkdir(parents=True, exist_ok=True)
    pipeline = SkillCandidatePipeline(journal=journal, proposal_manager=manager)

    status = pipeline.status()

    assert status["kind"] == "skill_candidate_pipeline_status"
    assert status["observe_only"] is True
    assert "activate skill" in status["blocked_actions"]


def test_skill_candidate_dry_run_does_not_write_proposal(tmp_path: Path):
    journal = TaskJournal(tmp_path / "memory" / "agent_journal.jsonl")
    _append_tool(journal, "echo")
    _append_tool(journal, "echo")
    manager = SkillProposalManager()
    manager.root = tmp_path / "skill_proposals"
    manager.root.mkdir(parents=True, exist_ok=True)
    pipeline = SkillCandidatePipeline(journal=journal, proposal_manager=manager)

    result = pipeline.run_once(force=True, dry_run=True)

    assert result["status"] == "planned"
    assert result["auto_changes_made"] is False
    assert not list((tmp_path / "skill_proposals").glob("skill_*"))


def test_skill_candidate_run_creates_reviewable_proposal_only(tmp_path: Path):
    journal = TaskJournal(tmp_path / "memory" / "agent_journal.jsonl")
    _append_tool(journal, "echo")
    _append_tool(journal, "echo")
    manager = SkillProposalManager()
    manager.root = tmp_path / "skill_proposals"
    manager.root.mkdir(parents=True, exist_ok=True)
    manager.detector.journal = journal
    pipeline = SkillCandidatePipeline(journal=journal, proposal_manager=manager)

    result = pipeline.run_once(force=True, dry_run=False, name="Echo Routine")

    assert result["status"] == "completed"
    assert result["observe_only"] is True
    assert result["activated"] is False
    proposal_dir = Path(result["proposal"]["proposal_dir"])
    assert (proposal_dir / "proposal.json").exists()
    proposal = json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    assert proposal["skill"]["name"] == "Echo Routine"
    assert proposal["status"] in {"VALIDATED", "FAILED"}
