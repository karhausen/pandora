from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import LLM_CONFIG_LOCAL_FILE
from .llm_config import LLMConfig
from .model_router import DEFAULT_MODEL_ROUTES, ModelRouter


ALLOWED_PURPOSES = set(DEFAULT_MODEL_ROUTES.keys()) | {
    "night_mode",
    "maintenance",
    "skill_generation",
    "skill_review",
}


@dataclass(frozen=True)
class RoutingValidation:
    ok: bool
    issues: list[str]
    warnings: list[str]


class LLMRoutingEditorService:
    """Controlled editor for Pandora's model routing rules.

    The editor writes only non-secret routing overrides to
    config/llm/llm_config.local.json. It never writes provider credentials,
    base URLs, API keys or other secret-bearing fields.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        local_path: Path = LLM_CONFIG_LOCAL_FILE,
        audit_path: Path | None = None,
    ):
        self.config = config or LLMConfig(local_path=local_path)
        self.local_path = Path(local_path)
        self.audit_path = audit_path or self.local_path.parent / "llm_routing_audit.jsonl"

    def status(self) -> dict[str, Any]:
        cfg = self.config.get()
        return {
            "kind": "llm_routing_editor_status",
            "editable": True,
            "active_profile": cfg.get("active_profile"),
            "local_override_file": str(self.local_path),
            "local_override_exists": self.local_path.exists(),
            "audit_file": str(self.audit_path),
            "purpose_count": len(self.available_purposes()),
            "provider_count": len(self.available_providers()),
            "guardrails": [
                "Routing-Änderungen schreiben nur nicht-geheime Local-Overrides.",
                "Provider müssen aus der bestehenden Providerliste stammen oder cloud_expert sein.",
                "Secrets, Base-URLs und API-Keys werden nicht über diesen Editor gespeichert.",
                "Jede Änderung erzeugt einen Audit-Eintrag und ein Backup der vorherigen Local-Config.",
            ],
        }

    def available_purposes(self) -> list[str]:
        cfg = self.config.get()
        return sorted(ALLOWED_PURPOSES | set(cfg.get("model_routes", {})) | set(cfg.get("routing", {})))

    def available_providers(self) -> list[str]:
        cfg = self.config.get()
        providers = set((cfg.get("providers") or {}).keys())
        aliases = set((cfg.get("provider_aliases") or {}).keys())
        allowed = providers | aliases | {"cloud_expert"}
        return sorted(allowed)

    def routes(self) -> dict[str, Any]:
        router = ModelRouter(config=self.config)
        raw_cfg = self.config.get()
        routes = []
        for purpose in self.available_purposes():
            resolved = router.route(purpose).model_dump(mode="json")
            configured = (raw_cfg.get("model_routes") or {}).get(purpose) or (raw_cfg.get("routing") or {}).get(purpose) or {}
            routes.append({
                "purpose": purpose,
                "provider": configured.get("provider") or resolved.get("provider_name"),
                "model": configured.get("model") or resolved.get("model"),
                "reason": configured.get("reason") or resolved.get("reason"),
                "resolved": resolved,
                "editable": purpose in self.available_purposes(),
            })
        return {
            "kind": "llm_routing_rules",
            "active_profile": raw_cfg.get("active_profile"),
            "providers": self.available_providers(),
            "purposes": self.available_purposes(),
            "routes": routes,
        }

    def preview_update(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = self._normalize_updates(updates)
        validation = self._validate_updates(normalized)
        current_routes = self.routes()["routes"]
        next_routes = []
        if validation.ok:
            next_cfg = self._merged_local_with_updates(normalized)
            preview_config = LLMConfig(
                path=self.config.path,
                local_path=self._write_temp_config(next_cfg),
                template_path=self.config.template_path,
                env_path=self.config.env_path,
            )
            next_router = ModelRouter(config=preview_config)
            for purpose in self.available_purposes():
                next_routes.append(next_router.route(purpose).model_dump(mode="json"))
            # Remove temp file immediately; preview should not persist anything.
            try:
                preview_config.local_path.unlink(missing_ok=True)
            except Exception:
                pass
        return {
            "kind": "llm_routing_update_preview",
            "ok": validation.ok,
            "issues": validation.issues,
            "warnings": validation.warnings,
            "updates": normalized,
            "current_routes": current_routes,
            "next_routes": next_routes,
            "will_write": False,
            "secret_safe": True,
        }

    def apply_update(self, updates: list[dict[str, Any]], actor: str = "user-gui") -> dict[str, Any]:
        normalized = self._normalize_updates(updates)
        validation = self._validate_updates(normalized)
        if not validation.ok:
            return {
                "success": False,
                "kind": "llm_routing_update",
                "issues": validation.issues,
                "warnings": validation.warnings,
                "updates": normalized,
            }
        before = self._read_local()
        after = self._merge_routes(before, normalized)
        backup = self._backup_local()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(json.dumps(after, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.config = LLMConfig(local_path=self.local_path)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "actor": actor,
            "action": "apply_routing_update",
            "updates": normalized,
            "backup": str(backup) if backup else None,
            "active_profile": self.config.get().get("active_profile"),
        }
        self._append_audit(event)
        return {
            "success": True,
            "kind": "llm_routing_update",
            "updates": normalized,
            "warnings": validation.warnings,
            "backup": str(backup) if backup else None,
            "routes": self.routes(),
            "audit_event": event,
        }

    def audit(self, limit: int = 50) -> dict[str, Any]:
        if not self.audit_path.exists():
            return {"kind": "llm_routing_audit", "events": [], "count": 0}
        events = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"raw": line, "parse_error": True})
        events = events[-max(1, int(limit)):]
        return {"kind": "llm_routing_audit", "events": events, "count": len(events)}

    def rollback(self, backup_path: str | None = None) -> dict[str, Any]:
        backups = sorted(self.local_path.parent.glob("llm_config.local.backup-*.json"))
        source = Path(backup_path) if backup_path else (backups[-1] if backups else None)
        if not source or not source.exists() or source.parent != self.local_path.parent:
            return {"success": False, "error": "No valid routing backup found."}
        shutil.copy2(source, self.local_path)
        self.config = LLMConfig(local_path=self.local_path)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "actor": "user-gui",
            "action": "rollback_routing_update",
            "backup": str(source),
        }
        self._append_audit(event)
        return {"success": True, "restored_from": str(source), "routes": self.routes(), "audit_event": event}

    def _normalize_updates(self, updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for item in updates or []:
            purpose = str(item.get("purpose", "")).strip()
            provider = str(item.get("provider", "")).strip()
            model = str(item.get("model", "")).strip()
            reason = str(item.get("reason", "")).strip()
            route: dict[str, Any] = {"purpose": purpose, "provider": provider}
            if model:
                route["model"] = model
            if reason:
                route["reason"] = reason[:300]
            for forbidden_key in ("api_key", "token", "password", "secret", "base_url"):
                if forbidden_key in item:
                    route[forbidden_key] = "<blocked>"
            normalized.append(route)
        return normalized

    def _validate_updates(self, updates: list[dict[str, Any]]) -> RoutingValidation:
        issues: list[str] = []
        warnings: list[str] = []
        providers = set(self.available_providers())
        purposes = set(self.available_purposes())
        if not updates:
            issues.append("No routing updates provided.")
        for item in updates:
            purpose = item.get("purpose")
            provider = item.get("provider")
            if not purpose or purpose not in purposes:
                issues.append(f"Unknown or unsupported purpose: {purpose}")
            if not provider or provider not in providers:
                issues.append(f"Unknown provider for {purpose}: {provider}")
            if provider == "company_llm":
                warnings.append(f"{purpose}: company_llm may require company network and ENV values.")
            if provider in {"openai", "cloud_expert"}:
                warnings.append(f"{purpose}: cloud routing can create external API usage/costs.")
            for forbidden_key in ("api_key", "token", "password", "secret", "base_url"):
                if forbidden_key in item:
                    issues.append(f"Forbidden secret-bearing field in update: {forbidden_key}")
        return RoutingValidation(ok=not issues, issues=issues, warnings=warnings)

    def _read_local(self) -> dict[str, Any]:
        if not self.local_path.exists():
            return {}
        try:
            return json.loads(self.local_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _merge_routes(self, local_data: dict[str, Any], updates: list[dict[str, Any]]) -> dict[str, Any]:
        data = dict(local_data)
        model_routes = dict(data.get("model_routes") or {})
        for item in updates:
            purpose = item["purpose"]
            route = dict(model_routes.get(purpose) or {})
            route["provider"] = item["provider"]
            if "model" in item:
                route["model"] = item["model"]
            else:
                route.pop("model", None)
            route["reason"] = item.get("reason") or f"User configured routing for {purpose}."
            model_routes[purpose] = route
        data["model_routes"] = model_routes
        return data

    def _merged_local_with_updates(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        return self._merge_routes(self._read_local(), updates)

    def _backup_local(self) -> Path | None:
        if not self.local_path.exists():
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = self.local_path.with_name(f"llm_config.local.backup-{stamp}.json")
        shutil.copy2(self.local_path, backup)
        return backup

    def _write_temp_config(self, data: dict[str, Any]) -> Path:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.local_path.with_name("llm_config.local.preview.json")
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return temp

    def _append_audit(self, event: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
