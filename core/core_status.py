from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ROOT_DIR, MEMORY_DIR, PROTECTED_CORE_FILES
from .llm_profile_manager import LLMProfileManager
from .model_router import ModelRouter
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry

PANDORA_CORE_VERSION = "mvp-22.9.3-documentation-cleanup"


@dataclass(frozen=True)
class ComponentCheck:
    name: str
    ok: bool
    message: str
    details: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        data = {"name": self.name, "ok": self.ok, "message": self.message}
        if self.details is not None:
            data["details"] = self.details
        return data


class CoreStatusService:
    """Single source of truth for Pandora's protected control-core status.

    This service is intentionally small. It answers: is the core alive, which
    profile is active, are registries reachable, and are protected files present?
    Growth features may depend on this service, but this service must not depend
    on generated tools or experimental skills.
    """

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir

    def status(self) -> dict[str, Any]:
        checks = [
            self._check_python(),
            self._check_paths(),
            self._check_profiles(),
            self._check_model_routes(),
            self._check_tool_registry(),
            self._check_skill_registry(),
            self._check_protected_core_files(),
        ]
        return {
            "status": "ok" if all(c.ok for c in checks) else "degraded",
            "version": PANDORA_CORE_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "role": "stable control core",
            "checks": [c.model_dump() for c in checks],
            "principles": {
                "core_is_protected": True,
                "growth_layer": ["tools", "skills", "workflows", "memory"],
                "core_changes_need_review": True,
                "secrets_in_environment_only": True,
            },
        }

    def _check_python(self) -> ComponentCheck:
        ok = sys.version_info >= (3, 12)
        return ComponentCheck("python", ok, f"Python {sys.version.split()[0]}")

    def _check_paths(self) -> ComponentCheck:
        required = [self.root_dir / "core", self.root_dir / "tools", self.root_dir / "skills", MEMORY_DIR]
        missing = [str(p.relative_to(self.root_dir)) for p in required if not p.exists()]
        return ComponentCheck("paths", not missing, "required folders present" if not missing else "missing folders", {"missing": missing})

    def _check_profiles(self) -> ComponentCheck:
        try:
            status = LLMProfileManager().status()
            return ComponentCheck("llm_profiles", bool(status.get("active_profile")), "profile system reachable", status)
        except Exception as exc:
            return ComponentCheck("llm_profiles", False, f"{type(exc).__name__}: {exc}")

    def _check_model_routes(self) -> ComponentCheck:
        try:
            routes = ModelRouter().all_routes()
            required = {"chat", "planning", "tool_generation", "tool_design", "core_review"}
            missing = sorted(required - set(routes))
            return ComponentCheck("model_router", not missing, "required routes present" if not missing else "missing routes", {"missing": missing})
        except Exception as exc:
            return ComponentCheck("model_router", False, f"{type(exc).__name__}: {exc}")

    def _check_tool_registry(self) -> ComponentCheck:
        try:
            registry = ToolRegistry()
            discovered = registry.discover()
            return ComponentCheck("tool_registry", True, "reachable", {"registered": len(registry.list()), "discovered": discovered})
        except Exception as exc:
            return ComponentCheck("tool_registry", False, f"{type(exc).__name__}: {exc}")

    def _check_skill_registry(self) -> ComponentCheck:
        try:
            registry = SkillRegistry()
            discovered = registry.discover()
            return ComponentCheck("skill_registry", True, "reachable", {"registered": len(registry.list()), "discovered": discovered})
        except Exception as exc:
            return ComponentCheck("skill_registry", False, f"{type(exc).__name__}: {exc}")

    def _check_protected_core_files(self) -> ComponentCheck:
        core_dir = self.root_dir / "core"
        missing = sorted(name for name in PROTECTED_CORE_FILES if not (core_dir / name).exists())
        # Some protected files are planned, not present in older snapshots. This is degraded, not fatal.
        ok = not missing
        return ComponentCheck("protected_core_files", ok, "all protected files present" if ok else "some protected files are missing", {"missing": missing})
