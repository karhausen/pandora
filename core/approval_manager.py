from __future__ import annotations
import json
from datetime import datetime, UTC
from pathlib import Path

class ApprovalManager:
    def approve(self, proposal_dir: Path, reviewer: str = "user") -> dict:
        approval = {"approved": True, "reviewer": reviewer, "approved_at": datetime.now(UTC).isoformat()}
        (proposal_dir / "approval.json").write_text(json.dumps(approval, indent=2, ensure_ascii=False), encoding="utf-8")
        return approval

    def reject(self, proposal_dir: Path, reason: str, reviewer: str = "user") -> dict:
        rejection = {"approved": False, "reviewer": reviewer, "reason": reason, "rejected_at": datetime.now(UTC).isoformat()}
        (proposal_dir / "approval.json").write_text(json.dumps(rejection, indent=2, ensure_ascii=False), encoding="utf-8")
        return rejection

    def is_approved(self, proposal_dir: Path) -> bool:
        path = proposal_dir / "approval.json"
        if not path.exists():
            return False
        return bool(json.loads(path.read_text(encoding="utf-8")).get("approved"))
