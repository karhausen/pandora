from __future__ import annotations

from typing import Any

from .evolution_lifecycle import EvolutionLifecycle
from .genome import PandoraGenome
from .genome_rules import GENOME_RULES
from .genome_schema import REQUIRED_GENOME_SECTIONS


class GenomeValidator:
    def validate(self, genome: PandoraGenome) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        data = genome.as_dict()
        for section in REQUIRED_GENOME_SECTIONS:
            if section not in data or not isinstance(data.get(section), dict):
                issues.append({"level": "error", "code": "missing_section", "section": section})
        lifecycle = data.get("evolution_rules", {}).get("single_lifecycle", [])
        missing_lifecycle = [step for step in EvolutionLifecycle.ids() if step not in lifecycle]
        if missing_lifecycle:
            issues.append({"level": "error", "code": "incomplete_lifecycle", "missing": missing_lifecycle})
        boundaries = data.get("boundaries", {})
        if boundaries.get("core_direct_write") is not False:
            issues.append({"level": "error", "code": "unsafe_boundary", "field": "core_direct_write"})
        if boundaries.get("identity_auto_change") is not False:
            issues.append({"level": "error", "code": "unsafe_boundary", "field": "identity_auto_change"})
        if data.get("safety", {}).get("human_approval_required") is not True:
            issues.append({"level": "error", "code": "human_approval_missing"})
        if not GENOME_RULES:
            issues.append({"level": "error", "code": "missing_rules"})
        return {
            "kind": "genome_validation_result",
            "version": "28.4",
            "ok": not any(item["level"] == "error" for item in issues),
            "issue_count": len(issues),
            "issues": issues,
            "checks": {
                "schema_valid": not any(item.get("code") == "missing_section" for item in issues),
                "lifecycle_valid": not any(item.get("code") == "incomplete_lifecycle" for item in issues),
                "rules_present": bool(GENOME_RULES),
                "human_approval_required": data.get("safety", {}).get("human_approval_required") is True,
                "core_direct_write_blocked": boundaries.get("core_direct_write") is False,
                "identity_auto_change_blocked": boundaries.get("identity_auto_change") is False,
            },
        }
