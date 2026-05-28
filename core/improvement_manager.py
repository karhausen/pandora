from __future__ import annotations
import json
import shutil
from datetime import datetime, UTC
from pathlib import Path
from .approval_manager import ApprovalManager
from .config import ROOT_DIR
from .patch_proposal import PatchProposalStore
from .regression_runner import RegressionRunner

class ImprovementManager:
    def __init__(self):
        self.store = PatchProposalStore()
        self.approvals = ApprovalManager()
        self.regression = RegressionRunner()

    def _copy_project(self, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        ignore = shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "proposals", "core_versions", "logs")
        shutil.copytree(ROOT_DIR, target, ignore=ignore)

    def propose_text_file_change(self, title: str, description: str, file_path: str, new_content: str, rationale: str | None = None) -> dict:
        return self.store.create(title, description, {file_path: new_content}, rationale=rationale).model_dump(mode="json")

    def propose_readme_note(self, title: str, note: str) -> dict:
        readme = ROOT_DIR / "README.md"
        current = readme.read_text(encoding="utf-8") if readme.exists() else ""
        new_content = current.rstrip() + "\n\n## " + title + "\n\n" + note.strip() + "\n"
        return self.propose_text_file_change(title, "Append a controlled improvement note to README.md.", "README.md", new_content, rationale="Safe demo proposal for the controlled self-improvement pipeline.")

    def list(self) -> list[dict]:
        return self.store.list()

    def show(self, proposal_id: str) -> dict:
        return self.store.load(proposal_id)

    def validate(self, proposal_id: str) -> dict:
        proposal_dir = self.store.proposal_dir(proposal_id)
        sandbox = proposal_dir / "sandbox"
        self._copy_project(sandbox)
        self.store.stage_to_snapshot(proposal_id, sandbox)
        validation = self.regression.run_all(sandbox)
        validation["validated_at"] = datetime.now(UTC).isoformat()
        (proposal_dir / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
        return validation

    def approve(self, proposal_id: str, reviewer: str = "user") -> dict:
        return self.approvals.approve(self.store.proposal_dir(proposal_id), reviewer=reviewer)

    def reject(self, proposal_id: str, reason: str, reviewer: str = "user") -> dict:
        return self.approvals.reject(self.store.proposal_dir(proposal_id), reason=reason, reviewer=reviewer)

    def prepare_snapshot(self, proposal_id: str) -> dict:
        proposal_dir = self.store.proposal_dir(proposal_id)
        if not self.approvals.is_approved(proposal_dir):
            return {"prepared": False, "error": "Proposal is not approved"}
        snapshot = proposal_dir / "approved_snapshot"
        self._copy_project(snapshot)
        applied = self.store.stage_to_snapshot(proposal_id, snapshot)
        return {"prepared": True, "snapshot": str(snapshot), **applied}
