from __future__ import annotations
import difflib
from pathlib import Path

class DiffManager:
    def create_unified_diff(self, original: str, modified: str, file_path: str) -> str:
        return "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))

    def apply_full_file_patch(self, root: Path, file_path: str, new_content: str) -> Path:
        target = (root / file_path).resolve()
        root_resolved = root.resolve()
        if root_resolved not in target.parents and target != root_resolved:
            raise ValueError(f"Target outside root: {file_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        return target
