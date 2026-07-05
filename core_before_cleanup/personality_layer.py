from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cognitive_identity import CognitiveIdentityService


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "system" / "personality.json"


@dataclass
class PersonalityLayerService:
    """Personality and prompt architecture layer for Pandora.

    MVP 28.1 separates *who Pandora is allowed to claim to be* from *how
    Pandora should communicate*. The service is read-only by default: it builds
    prompt packages and style contracts, but does not call an LLM, execute tools,
    write memory, approve proposals, or modify core files.
    """

    identity_service: CognitiveIdentityService | None = None
    config_path: Path = DEFAULT_CONFIG_PATH
    version: str = "28.1"
    fallback_profile: str = "balanced"
    _config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.identity_service = self.identity_service or CognitiveIdentityService(version="28.1")
        self._config = self._load_config()

    def status(self) -> dict[str, Any]:
        profiles = self._config.get("profiles", {})
        active = self.active_profile_name()
        return {
            "kind": "personality_layer_status",
            "ok": True,
            "mvp": self.version,
            "active_profile": active,
            "available_profiles": sorted(profiles.keys()),
            "outputs": ["profile", "style_contract", "prompt_package", "prompt_preview"],
            "guarantee": "Read-only prompt architecture. No LLM call, no execution, no persistence, no approval bypass.",
        }

    def active_profile_name(self) -> str:
        profiles = self._config.get("profiles", {})
        configured = self._config.get("active_profile") or self.fallback_profile
        return configured if configured in profiles else self.fallback_profile

    def profile(self, profile_name: str | None = None) -> dict[str, Any]:
        profiles = self._config.get("profiles", {})
        selected = profile_name or self.active_profile_name()
        if selected not in profiles:
            selected = self.fallback_profile
        profile = dict(profiles.get(selected, {}))
        return {
            "kind": "personality_profile",
            "mvp": self.version,
            "id": selected,
            "name": profile.get("name", selected),
            "tone": profile.get("tone", "klar, freundlich, ehrlich"),
            "verbosity": profile.get("verbosity", "mittel"),
            "style_rules": profile.get("style_rules", []),
            "source": str(self.config_path),
        }

    def style_contract(self, profile_name: str | None = None) -> dict[str, Any]:
        profile = self.profile(profile_name)
        identity = self.identity_service.identity_card()
        boundaries = self.identity_service.capability_boundaries()
        return {
            "kind": "style_contract",
            "mvp": self.version,
            "profile": profile,
            "identity_constraints": {
                "name": identity.get("name"),
                "system_type": identity.get("system_type"),
                "must_not_claim": identity.get("core_identity", {}).get("is_not", []),
            },
            "truthfulness_rules": boundaries.get("truthfulness_rules", []),
            "response_rules": [
                "Use the selected tone without weakening safety boundaries.",
                "State uncertainty, missing tests, and missing live data clearly.",
                "Separate suggestion, approval, and execution whenever a change is requested.",
                "Do not imply that this prompt layer executed any action.",
            ],
        }

    def prompt_package(self, request: str, *, profile_name: str | None = None, output_contract: str | None = None) -> dict[str, Any]:
        profile = self.profile(profile_name)
        identity = self.identity_service.identity_card()
        boundaries = self.identity_service.capability_boundaries()
        architecture = self._config.get("prompt_architecture", {})
        contract = output_contract or architecture.get("default_output_contract") or "Answer clearly and honestly."
        layers = [
            {
                "layer": "identity",
                "content": f"You are {identity.get('name')}, {identity.get('system_type')}. Mission: {identity.get('mission')}",
            },
            {
                "layer": "personality",
                "content": f"Tone: {profile.get('tone')}. Verbosity: {profile.get('verbosity')}. Rules: " + "; ".join(profile.get("style_rules", [])),
            },
            {
                "layer": "capability_boundaries",
                "content": "Can do: " + "; ".join(boundaries.get("can_do", [])) + " | Must stop before: " + "; ".join(boundaries.get("must_ask_or_stop_before", [])),
            },
            {
                "layer": "task_context",
                "content": request,
            },
            {
                "layer": "output_contract",
                "content": contract,
            },
            {
                "layer": "safety_gate",
                "content": "No risky action, external side effect, proposal execution, config change, or core modification without explicit approval and validation.",
            },
        ]
        return {
            "kind": "prompt_package",
            "mvp": self.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "architecture": {
                "layers": architecture.get("layers", [layer["layer"] for layer in layers]),
                "read_only": True,
                "llm_called": False,
            },
            "request": request,
            "layers": layers,
            "prompt_preview": self._render_prompt(layers),
            "trace": {
                "config_path": str(self.config_path),
                "identity_mvp": identity.get("mvp"),
                "execution_allowed_by_this_service": False,
            },
        }

    def prompt_preview(self, request: str, *, profile_name: str | None = None, output_contract: str | None = None) -> dict[str, Any]:
        package = self.prompt_package(request, profile_name=profile_name, output_contract=output_contract)
        return {
            "kind": "prompt_preview",
            "mvp": self.version,
            "profile_id": package["profile"].get("id"),
            "prompt": package["prompt_preview"],
            "read_only": True,
        }

    def _render_prompt(self, layers: list[dict[str, str]]) -> str:
        rendered: list[str] = []
        for layer in layers:
            rendered.append(f"[{layer.get('layer')}]")
            rendered.append(str(layer.get("content", "")).strip())
        return "\n".join(rendered).strip()

    def _load_config(self) -> dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "mvp": self.version,
                "active_profile": self.fallback_profile,
                "profiles": {
                    self.fallback_profile: {
                        "name": "Balanced",
                        "tone": "klar, freundlich, ehrlich, praxisnah",
                        "verbosity": "mittel",
                        "style_rules": ["honest limits", "clear next steps", "approval before risky changes"],
                    }
                },
                "prompt_architecture": {"layers": ["identity", "personality", "capability_boundaries", "task_context", "output_contract", "safety_gate"]},
            }
        except json.JSONDecodeError as exc:
            return {
                "mvp": self.version,
                "active_profile": self.fallback_profile,
                "profiles": {
                    self.fallback_profile: {
                        "name": "Balanced",
                        "tone": "klar, freundlich, ehrlich",
                        "verbosity": "mittel",
                        "style_rules": [f"personality config invalid: {exc}"],
                    }
                },
            }
