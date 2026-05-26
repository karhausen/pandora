from __future__ import annotations

import json
from datetime import datetime, UTC
from .config import TOOLS_DIR, PROPOSALS_DIR
from .models import ToolSpec
from .security import ToolSecurityValidator
from .tool_registry import ToolRegistry
from .tool_validator import ToolValidator
from .tool_tester import ToolTester


class ToolLifecycleManager:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.validator = ToolValidator()
        self.tester = ToolTester()
        self.security = ToolSecurityValidator()

    def propose_and_activate(self, spec: ToolSpec) -> dict:
        proposal_dir = PROPOSALS_DIR / "tools" / spec.id
        proposal_dir.mkdir(parents=True, exist_ok=True)
        proposal = {
            "id": spec.id,
            "capability": spec.capability,
            "description": spec.description,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "generated",
        }
        (proposal_dir / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        (proposal_dir / f"{spec.id}.py").write_text(spec.code, encoding="utf-8")

        validation = self.validator.validate(spec)
        (proposal_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        if not validation["valid"]:
            return {"activated": False, "stage": "validation", "errors": validation["errors"]}

        target = TOOLS_DIR / f"{spec.id}.py"
        self.security.validate_target_path(target, TOOLS_DIR)
        target.write_text(spec.code, encoding="utf-8")

        tests = self.tester.run_tests(spec, TOOLS_DIR)
        (proposal_dir / "tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
        if not tests["passed"]:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            return {"activated": False, "stage": "tests", "errors": tests["errors"]}

        self.registry.discover()
        return {"activated": True, "tool_id": spec.id, "proposal_dir": str(proposal_dir)}
