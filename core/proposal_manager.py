from __future__ import annotations
from .config import PROPOSALS_DIR
class ProposalManager:
    def list_proposals(self):
        return [{'path': str(p), 'type': p.parent.parent.name if p.parent.parent else 'unknown'} for p in PROPOSALS_DIR.rglob('*.json')]
