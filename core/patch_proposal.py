from __future__ import annotations
import json
import uuid
from datetime import datetime, UTC
from pathlib import Path
from .code_review import CodeReview
from .config import ROOT_DIR, IMPROVEMENTS_DIR
from .diff_manager import DiffManager
from .models import ImprovementProposal, ImprovementRisk, ImprovementStatus

class PatchProposalStore:
    def __init__(self, root: Path = IMPROVEMENTS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def proposal_dir(self, proposal_id: str) -> Path:
        return self.root / proposal_id

    def create(self, title: str, description: str, changes: dict[str, str], rationale: str | None = None) -> ImprovementProposal:
        proposal_id = f"imp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        proposal_dir = self.proposal_dir(proposal_id)
        proposal_dir.mkdir(parents=True, exist_ok=False)
        changes_dir = proposal_dir / "changes"
        changes_dir.mkdir()
        review = CodeReview().review_many(changes)
        diff_manager = DiffManager()
        for file_path, new_content in changes.items():
            original_path = ROOT_DIR / file_path
            original = original_path.read_text(encoding="utf-8") if original_path.exists() else ""
            safe_name = file_path.replace("/", "__").replace("\\", "__")
            (changes_dir / f"{safe_name}.new").write_text(new_content, encoding="utf-8")
            (changes_dir / f"{safe_name}.diff").write_text(diff_manager.create_unified_diff(original, new_content, file_path), encoding="utf-8")
        proposal = ImprovementProposal(id=proposal_id, title=title, description=description, status=ImprovementStatus.PROPOSED, risk=ImprovementRisk(review["risk"]), target_files=list(changes.keys()), created_at=datetime.now(UTC).isoformat(), rationale=rationale)
        (proposal_dir / "proposal.json").write_text(json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        (proposal_dir / "review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        (proposal_dir / "changes.json").write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
        return proposal

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self.root.glob("imp_*"), reverse=True):
            p = path / "proposal.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                data["path"] = str(path)
                out.append(data)
        return out

    def load(self, proposal_id: str) -> dict:
        path = self.proposal_dir(proposal_id)
        if not path.exists():
            raise FileNotFoundError(proposal_id)
        result = {"path": str(path)}
        for name in ["proposal.json", "review.json", "changes.json", "validation.json", "approval.json"]:
            p = path / name
            if p.exists():
                result[name.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))
        return result

    def stage_to_snapshot(self, proposal_id: str, snapshot_root: Path) -> dict:
        changes = self.load(proposal_id)["changes"]
        applied = []
        for file_path, new_content in changes.items():
            target = snapshot_root / file_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_content, encoding="utf-8")
            applied.append(file_path)
        return {"applied": applied, "snapshot_root": str(snapshot_root)}
