from __future__ import annotations

import hashlib
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


class BuildManifest:
    """Create a small, reproducible release manifest."""

    def __init__(self, root_dir: Path | str = ".") -> None:
        self.root_dir = Path(root_dir).resolve()

    def create(self, *, version: str, based_on: str | None = None, test_summary: dict[str, Any] | None = None, audit: dict[str, Any] | None = None) -> dict[str, Any]:
        files = [p for p in self.root_dir.rglob("*") if p.is_file() and ".git" not in p.parts]
        digest = hashlib.sha256()
        for path in sorted(files):
            rel = str(path.relative_to(self.root_dir)).replace("\\", "/")
            digest.update(rel.encode("utf-8"))
            try:
                digest.update(path.read_bytes())
            except OSError:
                pass
        manifest = {
            "kind": "pandora_release_manifest",
            "version": version,
            "based_on": based_on,
            "build_time": datetime.now(UTC).isoformat(),
            "file_count": len(files),
            "content_sha256": digest.hexdigest(),
            "tests": test_summary or {},
            "audit": audit or {},
            "complete_project_release": True,
        }
        (self.root_dir / "release.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest
