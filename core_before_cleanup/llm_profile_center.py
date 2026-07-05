from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_config import LLMConfig
from .llm_profile_manager import LLMProfileManager
from .model_router import ModelRouter


@dataclass(frozen=True)
class ProfileDecision:
    allowed: bool
    reason: str


class LLMProfileCenterService:
    """Read-only first GUI service for LLM profiles and routing.

    The service intentionally exposes only sanitized configuration. API keys,
    raw environment values and local secret files must never be returned to the
    web UI. Profile switching is allowed because it only writes the selected
    profile name to the local override file.
    """

    def __init__(self, manager: LLMProfileManager | None = None, config: LLMConfig | None = None):
        self.manager = manager or LLMProfileManager()
        self.config = config or LLMConfig()

    def dashboard(self) -> dict[str, Any]:
        status = self.manager.status()
        cfg = self.config.public_config()
        routes = ModelRouter().all_routes()
        providers = self.providers()
        active_profile = status.get("active_profile")
        return {
            "kind": "llm_profile_center_dashboard",
            "title": "LLM & Profile Center",
            "active_profile": active_profile,
            "available_profiles": status.get("available_profiles", []),
            "cloud_expert_provider": status.get("cloud_expert_provider"),
            "security": status.get("security"),
            "routes": routes,
            "providers": providers["providers"],
            "profile_purpose": {
                "private": "Privates Profil: lokale Modelle plus private Cloud-LLM, falls freigegeben.",
                "company": "Firmenprofil: Company-LLM und interne Endpunkte, Secrets nur per Environment.",
            },
            "guardrails": [
                "Secrets werden nicht in der GUI angezeigt.",
                "Profile-Umschaltung speichert nur den Profilnamen lokal.",
                "Live-Smoke-Tests bleiben explizit und werden nicht automatisch ausgeführt.",
            ],
            "config_summary": cfg,
        }

    def profiles(self) -> dict[str, Any]:
        status = self.manager.status()
        cfg = self.config.get()
        profiles = cfg.get("profiles") or {}
        result = []
        for name in sorted(profiles.keys()):
            profile_cfg = profiles.get(name) or {}
            cloud_provider = profile_cfg.get("cloud_expert")
            result.append({
                "name": name,
                "active": name == status.get("active_profile"),
                "cloud_expert": cloud_provider,
                "cloud_provider_status": self.manager.provider_status(cloud_provider) if cloud_provider else None,
                "description": self._profile_description(name),
            })
        return {"kind": "llm_profiles", "active_profile": status.get("active_profile"), "profiles": result}

    def set_profile(self, profile: str) -> dict[str, Any]:
        decision = self._profile_change_allowed(profile)
        if not decision.allowed:
            return {"success": False, "error": decision.reason, "requested_profile": profile}
        result = self.manager.set_profile(profile)
        result["kind"] = "llm_profile_change"
        result["safety_note"] = "Only the active profile name was changed; no secrets were written."
        return result

    def providers(self) -> dict[str, Any]:
        cfg = self.config.get()
        providers_cfg = cfg.get("providers") or {}
        providers = []
        for name in sorted(providers_cfg.keys()):
            providers.append(self.manager.provider_status(name))
        return {"kind": "llm_providers", "providers": providers}

    def routes(self) -> dict[str, Any]:
        routes = ModelRouter().all_routes()
        return {"kind": "llm_routes", "routes": routes}

    def smoke_preview(self, provider: str = "cloud_expert") -> dict[str, Any]:
        """Non-live smoke preview for the GUI.

        Live calls can spend money or access company networks, so the GUI gets a
        safe preview endpoint first. A future admin-only action can add explicit
        live execution with a confirmation dialog.
        """
        return self.manager.smoke(provider=provider, live=False)

    def _profile_change_allowed(self, profile: str) -> ProfileDecision:
        available = set(self.manager.status().get("available_profiles") or [])
        if profile not in available:
            return ProfileDecision(False, f"Unknown profile: {profile}")
        return ProfileDecision(True, "profile exists")

    def _profile_description(self, name: str) -> str:
        if name == "private":
            return "Private Nutzung: lokale Modelle und private Cloud-LLM."
        if name == "company":
            return "Firmennetz: Company-LLM und interne Endpunkte."
        return "Benutzerdefiniertes Profil."
