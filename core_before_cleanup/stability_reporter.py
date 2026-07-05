from __future__ import annotations

import json
from pathlib import Path

from .config import CORE_VERSIONS_DIR, MEMORY_DIR, STABILITY_REPORT_FILE


class StabilityReporter:
    def snapshot_summary(self) -> dict:
        versions = []
        total_size = 0

        if CORE_VERSIONS_DIR.exists():
            for path in CORE_VERSIONS_DIR.iterdir():
                if not path.is_dir():
                    continue
                size = self._dir_size(path)
                total_size += size
                versions.append({
                    "version_id": path.name,
                    "size_bytes": size,
                    "file_count": sum(1 for p in path.rglob("*") if p.is_file()),
                })

        return {
            "count": len(versions),
            "total_size_bytes": total_size,
            "versions": sorted(versions, key=lambda x: x["version_id"]),
        }

    def memory_summary(self) -> dict:
        files = []
        total_size = 0
        if MEMORY_DIR.exists():
            for path in MEMORY_DIR.glob("*"):
                if path.is_file():
                    size = path.stat().st_size
                    total_size += size
                    files.append({"name": path.name, "size_bytes": size})
        return {"total_size_bytes": total_size, "files": sorted(files, key=lambda x: x["name"])}

    def report(self) -> dict:
        report = {
            "snapshots": self.snapshot_summary(),
            "memory": self.memory_summary(),
        }
        STABILITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        STABILITY_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _dir_size(self, path: Path) -> int:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
