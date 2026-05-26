from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, UTC
from pathlib import Path

from .config import (
    ROOT_DIR, CORE_VERSION_STORE, CORE_VERSION_MANIFEST,
    ACTIVE_VERSION_FILE, STABLE_VERSION_FILE
)
from .models import CoreVersionMeta, CoreVersionStatus


EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "core_versions", ".pytest_cache"}
SNAPSHOT_DIRS = ["core", "tools", "skills", "tests"]
SNAPSHOT_FILES = ["main.py", "requirements.txt", "README.md"]


class VersionManager:
    def __init__(self):
        CORE_VERSION_STORE.mkdir(parents=True, exist_ok=True)
        CORE_VERSION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        if not CORE_VERSION_MANIFEST.exists():
            CORE_VERSION_MANIFEST.write_text("{}", encoding="utf-8")

    def manifest(self) -> dict[str, dict]:
        return json.loads(CORE_VERSION_MANIFEST.read_text(encoding="utf-8"))

    def save_manifest(self, data: dict[str, dict]) -> None:
        CORE_VERSION_MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_snapshot(self, version_id: str | None = None) -> CoreVersionMeta:
        version_id = version_id or f"core_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        build_id = str(uuid.uuid4())
        target = CORE_VERSION_STORE / version_id
        if target.exists():
            raise FileExistsError(f"Version already exists: {version_id}")
        target.mkdir(parents=True)

        for d in SNAPSHOT_DIRS:
            src = ROOT_DIR / d
            if src.exists():
                shutil.copytree(src, target / d, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))

        for f in SNAPSHOT_FILES:
            src = ROOT_DIR / f
            if src.exists():
                shutil.copy2(src, target / f)

        meta = CoreVersionMeta(
            version_id=version_id,
            build_id=build_id,
            created_at=datetime.now(UTC).isoformat(),
            status=CoreVersionStatus.CREATED,
            path=str(target),
            rollback_target=self.get_stable_version() or self.get_active_version(),
        )
        self._write_version_meta(meta)
        manifest = self.manifest()
        manifest[version_id] = meta.model_dump(mode="json")
        self.save_manifest(manifest)
        return meta

    def _write_version_meta(self, meta: CoreVersionMeta) -> None:
        path = Path(meta.path) / "version.json"
        path.write_text(json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    def update_status(self, version_id: str, status: CoreVersionStatus, **fields) -> CoreVersionMeta:
        manifest = self.manifest()
        if version_id not in manifest:
            raise KeyError(f"Unknown version: {version_id}")
        data = manifest[version_id]
        data["status"] = status.value
        data.update(fields)
        meta = CoreVersionMeta.model_validate(data)
        manifest[version_id] = meta.model_dump(mode="json")
        self.save_manifest(manifest)
        self._write_version_meta(meta)
        return meta

    def list_versions(self) -> list[CoreVersionMeta]:
        return [CoreVersionMeta.model_validate(v) for v in self.manifest().values()]

    def get_version(self, version_id: str) -> CoreVersionMeta | None:
        data = self.manifest().get(version_id)
        return CoreVersionMeta.model_validate(data) if data else None

    def set_active_version(self, version_id: str) -> None:
        ACTIVE_VERSION_FILE.write_text(version_id, encoding="utf-8")

    def set_stable_version(self, version_id: str) -> None:
        STABLE_VERSION_FILE.write_text(version_id, encoding="utf-8")

    def get_active_version(self) -> str | None:
        return ACTIVE_VERSION_FILE.read_text(encoding="utf-8").strip() if ACTIVE_VERSION_FILE.exists() else None

    def get_stable_version(self) -> str | None:
        return STABLE_VERSION_FILE.read_text(encoding="utf-8").strip() if STABLE_VERSION_FILE.exists() else None
