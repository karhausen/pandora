from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from core.capability_event_log import CapabilityEventLog
from core.capability_gap_pipeline import CapabilityGapPipeline
from core.task_journal import TaskJournal


def make_pipeline(tmp_path: Path) -> CapabilityGapPipeline:
    event_log = CapabilityEventLog(tmp_path / "capability_events.jsonl")
    journal = TaskJournal(tmp_path / "agent_journal.jsonl")
    return CapabilityGapPipeline(event_log=event_log, journal=journal, output_dir=tmp_path / "proposals" / "capability_gaps")


def test_capability_gap_status_is_observe_only(tmp_path):
    pipeline = make_pipeline(tmp_path)
    status = pipeline.status()
    assert status["observe_only"] is True
    assert "generate tool code" in status["blocked_actions"]
    assert "create reviewable capability gap proposal JSON" in status["allowed_actions"]


def test_capability_gap_pipeline_skips_without_signals(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.run_once(min_signals=1, force=False)
    assert result["status"] == "skipped"
    assert result["auto_changes_made"] is False


def test_capability_gap_pipeline_dry_run_plans_without_writing(tmp_path):
    pipeline = make_pipeline(tmp_path)
    pipeline.event_log.append({"capability_gap": "weather forecast lookup", "task": "weather tomorrow"})
    result = pipeline.run_once(force=True, dry_run=True)
    assert result["status"] == "planned"
    assert result["activated"] is False
    assert not list((tmp_path / "proposals" / "capability_gaps").glob("*/proposal.json"))


def test_capability_gap_pipeline_writes_reviewable_proposal(tmp_path):
    pipeline = make_pipeline(tmp_path)
    pipeline.event_log.append({"capability_gap": "stock price lookup", "task": "price BASF"})
    pipeline.journal.append({"task": "Need stock price lookup", "reason": "missing capability: stock price lookup"})
    result = pipeline.run_once(force=True, dry_run=False)
    assert result["status"] == "completed"
    assert result["activated"] is False
    proposal_path = Path(result["written_to"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposal["kind"] == "capability_gap_proposal"
    assert proposal["review_required"] is True
    assert proposal["capability"]["priority"] == "high"
    assert "do not generate code automatically from this proposal" in proposal["blocked_actions"]


def test_capability_gap_cli_status_runs():
    completed = subprocess.run([sys.executable, "main.py", "capability-gap-status"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)
    data = json.loads(completed.stdout)
    assert data["kind"] == "capability_gap_pipeline_status"
    assert data["observe_only"] is True
