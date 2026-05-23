from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ImprovementManager:
    def __init__(self, proposal_dir: Path):
        self.proposal_dir = proposal_dir
        self.proposal_dir.mkdir(parents=True, exist_ok=True)

    def create_proposal(self, title: str, data: dict[str, Any]) -> Path:
        safe = title.lower().replace(" ", "_")[:40]
        path = self.proposal_dir / f"{int(time.time())}_{safe}.json"
        payload = {
            "title": title,
            "description": data.get("description", ""),
            "affected_files": data.get("affected_files", []),
            "risk_analysis": data.get("risk_analysis", "unknown"),
            "tests": data.get("tests", []),
            "rollback_plan": data.get("rollback_plan", "restore previous core version"),
            "created_at": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
