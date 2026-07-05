from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

from .release_audit import ReleaseAudit, RUNTIME_DIR_NAMES, RUNTIME_SUFFIXES
from .build_manifest import BuildManifest


class BuildManager:
    """Build complete Pandora release ZIPs, never patch-only archives."""

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()

    def clean_runtime_artifacts(self) -> dict[str, Any]:
        removed: list[str] = []
        for path in sorted(self.root_dir.rglob("*"), reverse=True):
            rel = str(path.relative_to(self.root_dir)).replace("\\", "/")
            if path.is_dir() and path.name in RUNTIME_DIR_NAMES:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(rel)
            elif path.is_file() and path.suffix in RUNTIME_SUFFIXES:
                try:
                    path.unlink()
                    removed.append(rel)
                except OSError:
                    pass
        return {"kind": "release_clean", "removed_count": len(removed), "removed": removed[:200]}

    def build_zip(self, *, version: str, output: Path | str, based_on: str | None = None, skip_audit: bool = False) -> dict[str, Any]:
        clean = self.clean_runtime_artifacts()
        audit = ReleaseAudit(self.root_dir).run()
        if not skip_audit and not audit.get("ok"):
            return {"kind": "release_build", "ok": False, "reason": "audit_failed", "audit": audit, "clean": clean}
        manifest = BuildManifest(self.root_dir).create(version=version, based_on=based_on, audit=audit)
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            base = self.root_dir.name
            for path in sorted(self.root_dir.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(self.root_dir)
                    zf.write(path, Path(base) / rel)
        return {"kind": "release_build", "ok": True, "output": str(output), "manifest": manifest, "audit": audit, "clean": clean}
