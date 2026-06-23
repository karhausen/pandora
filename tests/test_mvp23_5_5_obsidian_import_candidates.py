from __future__ import annotations

from pathlib import Path

from core.obsidian_import_candidates import ObsidianImportCandidateService
from core.obsidian_vault import ObsidianConfig, ObsidianVaultService
from core.proposal_review_inbox import ProposalReviewInbox


def _service(tmp_path: Path, *, cloud_allowed: bool = False) -> ObsidianImportCandidateService:
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "Funktechnik").mkdir()
    (vault_dir / "Funktechnik" / "Spektrumanalyse.md").write_text(
        "# Spektrumanalyse\n\nNotiz zu Funktechnik, Messgeraeten und Kalibrierung. #funktechnik #messtechnik\n\nSiehe [[Kalibrierung]].\n",
        encoding="utf-8",
    )
    config = ObsidianConfig(
        enabled=True,
        vault_path=vault_dir,
        inbox_dir="Pandora_Inbox",
        mode="read_write_inbox_only",
        cloud_allowed=cloud_allowed,
    )
    vault = ObsidianVaultService(root_dir=tmp_path, config=config)
    return ObsidianImportCandidateService(candidates_dir=tmp_path / "proposals" / "obsidian_import_candidates", vault=vault)


def test_obsidian_import_candidates_build_review_only(tmp_path: Path):
    svc = _service(tmp_path, cloud_allowed=False)
    report = svc.build(limit=10, write=True)
    assert report["candidate_count"] == 1
    candidate = report["candidates"][0]
    assert candidate["kind"] == "obsidian_import_candidate"
    assert candidate["target_area"] == "private_local_only"
    assert candidate["auto_import"] is False
    assert candidate["auto_write_knowledge"] is False
    assert "source_path" in candidate["proposed_metadata"]
    assert (tmp_path / "proposals" / "obsidian_import_candidates").exists()


def test_obsidian_import_candidates_list_show_decide(tmp_path: Path):
    svc = _service(tmp_path, cloud_allowed=True)
    report = svc.build(limit=10, write=True)
    candidate_id = report["candidates"][0]["id"]
    listing = svc.list_candidates(limit=10)
    assert listing["count"] == 1
    detail = svc.show(candidate_id)
    assert detail["found"] is True
    assert detail["source_preview"]["ok"] is True
    decision = svc.decide(candidate_id, decision="accepted_for_next_step", note="passt")
    assert decision["ok"] is True
    listing_after = svc.list_candidates(include_reviewed=True, limit=10)
    assert listing_after["candidates"][0]["status"] == "accepted_for_next_step"


def test_review_inbox_scans_obsidian_import_candidates(tmp_path: Path):
    svc = _service(tmp_path)
    svc.build(limit=10, write=True)
    inbox = ProposalReviewInbox(scan_dirs={"obsidian_import_candidate": tmp_path / "proposals" / "obsidian_import_candidates"})
    summary = inbox.summary(limit=10)
    assert summary["item_count"] == 1
    assert summary["items"][0]["category"] == "obsidian_import_candidate"
