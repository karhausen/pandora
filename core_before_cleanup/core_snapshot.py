from __future__ import annotations

import shutil
from datetime import datetime, UTC
from pathlib import Path

from .config import CORE_VERSIONS_DIR, ROOT_DIR


class CoreSnapshot:
    INCLUDE = [
        "core",
        "tools",
        "skills",
        "web",
        "docs",
        "tests",
        "main.py",
        "requirements.txt",
        "pytest.ini",
        "README.md",
    ]

    EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache"}

    def create(self, version_id: str | None = None) -> dict:
        version_id = version_id or datetime.now(UTC).strftime("core_%Y%m%d_%H%M%S")
        target = CORE_VERSIONS_DIR / version_id
        if target.exists():
            raise FileExistsError(version_id)

        target.mkdir(parents=True, exist_ok=False)

        copied = []
        for rel in self.INCLUDE:
            src = ROOT_DIR / rel
            if not src.exists():
                continue
            dst = target / rel
            if src.is_dir():
                shutil.copytree(src, dst, ignore=self._ignore)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied.append(rel)

        return {
            "version_id": version_id,
            "path": str(target),
            "copied": copied,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _ignore(self, dir_path, names):
        return {name for name in names if name in self.EXCLUDE_DIR_NAMES or name.endswith(".pyc")}
