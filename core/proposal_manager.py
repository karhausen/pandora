from __future__ import annotations
from .config import PROPOSALS_DIR

class ProposalManager:
    def list_proposals(self) -> list[dict]:
        results = []
        for path in PROPOSALS_DIR.rglob("*.json"):
            results.append({"path": str(path), "type": path.parent.parent.name if path.parent.parent else "unknown"})
        return results
