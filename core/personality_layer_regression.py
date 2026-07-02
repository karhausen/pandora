from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .personality_layer import PersonalityLayerService


@dataclass
class PersonalityLayerRegressionService:
    """Small deterministic regression checks for MVP 28.1."""

    service: PersonalityLayerService = field(default_factory=PersonalityLayerService)

    def status(self) -> dict[str, Any]:
        return {"kind": "personality_layer_regression_status", "ok": True, "mvp": "28.1"}

    def run(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        status = self.service.status()
        checks.append({"name": "status_ok", "ok": bool(status.get("ok"))})
        contract = self.service.style_contract()
        checks.append({"name": "style_contract_has_truthfulness", "ok": bool(contract.get("truthfulness_rules"))})
        package = self.service.prompt_package("Bitte baue einen sicheren nächsten Schritt.")
        layers = [layer.get("layer") for layer in package.get("layers", [])]
        required = ["identity", "personality", "capability_boundaries", "task_context", "output_contract", "safety_gate"]
        checks.append({"name": "all_prompt_layers_present", "ok": all(item in layers for item in required), "layers": layers})
        checks.append({"name": "read_only_no_execution", "ok": package.get("trace", {}).get("execution_allowed_by_this_service") is False})
        ok = all(check.get("ok") for check in checks)
        return {"kind": "personality_layer_regression", "mvp": "28.1", "ok": ok, "checks": checks}
