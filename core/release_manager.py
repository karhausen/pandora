from __future__ import annotations

from pathlib import Path
from typing import Any

from .build_manager import BuildManager
from .release_audit import ReleaseAudit
from .build_manifest import BuildManifest


class ReleaseManager:
    """User-facing release management facade for Operations."""

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.builder = BuildManager(self.root_dir)

    def status(self) -> dict[str, Any]:
        release_json = self.root_dir / "release.json"
        return {
            "kind": "release_manager_status",
            "root": str(self.root_dir),
            "release_json_exists": release_json.exists(),
            "complete_release_required": True,
            "available_actions": ["audit", "clean", "manifest", "build"],
        }

    def audit(self) -> dict[str, Any]:
        return ReleaseAudit(self.root_dir).run()

    def clean(self) -> dict[str, Any]:
        return self.builder.clean_runtime_artifacts()

    def manifest(self, *, version: str, based_on: str | None = None) -> dict[str, Any]:
        audit = self.audit()
        return BuildManifest(self.root_dir).create(version=version, based_on=based_on, audit=audit)

    def build(self, *, version: str, output: Path | str, based_on: str | None = None, skip_audit: bool = False) -> dict[str, Any]:
        return self.builder.build_zip(version=version, output=output, based_on=based_on, skip_audit=skip_audit)
