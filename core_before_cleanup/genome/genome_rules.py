from __future__ import annotations

from typing import Any

GENOME_RULES: list[dict[str, Any]] = [
    {"id": "core_no_direct_write", "title": "Core changes require Proposal + Review + Approval", "hard": True},
    {"id": "identity_locked", "title": "Identity must not be changed automatically", "hard": True},
    {"id": "personality_controlled", "title": "Personality changes require an EvolutionProposal", "hard": True},
    {"id": "runtime_no_genome_write", "title": "Runtime observations must not mutate the Genome directly", "hard": True},
    {"id": "llm_recommends_python_validates", "title": "LLM may recommend; Python validates and user decides", "hard": True},
    {"id": "tests_before_activation", "title": "Activation requires successful tests or explicit documented exception", "hard": True},
]

def rules_status() -> dict[str, Any]:
    return {
        "kind": "genome_rules_status",
        "version": "28.4",
        "rule_count": len(GENOME_RULES),
        "hard_rule_count": sum(1 for rule in GENOME_RULES if rule.get("hard")),
        "rules": GENOME_RULES,
    }
