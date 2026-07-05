from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .knowledge_governance import KnowledgeGovernanceService, VALID_PRIORITIES, VALID_VISIBILITIES
from .knowledge_metadata import parse_frontmatter, strip_frontmatter
from .user_knowledge_base import AREAS, UserKnowledgeBaseService

_ALLOWED_FILE_SUFFIXES = {".md", ".txt", ".json"}
_ALLOWED_NAME = re.compile(r"^[A-Za-z0-9ÄÖÜäöüß_.\- /]+$")


@dataclass
class KnowledgeEditorService:
    """Controlled write access for Pandora's user knowledge base.

    The editor only writes below user_knowledge/ and enforces area policy. It is
    deliberately separate from UserKnowledgeBaseService, which stays mostly
    read-oriented for search and context injection.
    """

    knowledge: UserKnowledgeBaseService = field(default_factory=UserKnowledgeBaseService)

    def status(self) -> dict[str, Any]:
        self.knowledge.ensure_structure()
        return {
            "kind": "knowledge_editor_status",
            "enabled": True,
            "root": str(self.knowledge.root_dir),
            "allowed_areas": [area.name for area in AREAS],
            "allowed_suffixes": sorted(_ALLOWED_FILE_SUFFIXES),
            "safety": {
                "root_locked": True,
                "path_traversal_blocked": True,
                "private_local_only_forces_cloud_allowed_false": True,
                "delete_requires_confirm": True,
            },
        }

    def tree(self) -> dict[str, Any]:
        self.knowledge.ensure_structure()
        areas: list[dict[str, Any]] = []
        for area in AREAS:
            root = self.knowledge.root_dir / area.directory
            folders: list[dict[str, Any]] = []
            files: list[dict[str, Any]] = []
            for path in sorted(root.rglob("*")):
                if path.name == ".gitkeep":
                    continue
                rel = path.relative_to(root).as_posix()
                if path.is_dir():
                    folders.append({"relative_path": rel, "name": path.name})
                elif path.is_file() and path.suffix.lower() in _ALLOWED_FILE_SUFFIXES:
                    stat = path.stat()
                    files.append({"relative_path": rel, "name": path.name, "type": path.suffix.lower().lstrip("."), "size_bytes": stat.st_size})
            areas.append({
                "name": area.name,
                "description": area.description,
                "cloud_allowed_default": area.cloud_allowed,
                "policy": area.policy,
                "folders": folders,
                "files": files,
            })
        return {"kind": "knowledge_editor_tree", "areas": areas}

    def metadata_template(self, *, area: str = "public", relative_path: str = "new-note.md") -> dict[str, Any]:
        area_def = self.knowledge._area(area)
        return {
            "title": Path(relative_path).stem.replace("_", " ").replace("-", " ").strip().title() or "Neue Notiz",
            "tags": [],
            "visibility": area_def.name,
            "cloud_allowed": area_def.cloud_allowed,
            "priority": "normal",
            "owner": "thomas",
            "last_reviewed": date.today().isoformat(),
            "summary": "",
        }

    def read_file(self, *, area: str, relative_path: str) -> dict[str, Any]:
        area_def = self.knowledge._area(area)
        root = self.knowledge.root_dir / area_def.directory
        path = self._file_path(root, relative_path, must_exist=True)
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_frontmatter(text) if path.suffix.lower() == ".md" else ({}, text)
        if path.suffix.lower() == ".md":
            metadata = self._enforce_policy(area_def.name, metadata or self.metadata_template(area=area_def.name, relative_path=relative_path))
        governance = KnowledgeGovernanceService(knowledge=self.knowledge).validate_text(text, area=area_def.name, relative_path=relative_path) if path.suffix.lower() == ".md" else {}
        return {
            "kind": "knowledge_editor_file",
            "found": True,
            "area": area_def.name,
            "relative_path": path.relative_to(root).as_posix(),
            "name": path.name,
            "type": path.suffix.lower().lstrip("."),
            "metadata": metadata,
            "body": body,
            "raw_text": text,
            "governance": governance,
        }

    def create_folder(self, *, area: str, relative_path: str) -> dict[str, Any]:
        area_def = self.knowledge._area(area)
        root = self.knowledge.root_dir / area_def.directory
        folder = self._folder_path(root, relative_path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").touch(exist_ok=True)
        return {"kind": "knowledge_editor_folder_created", "area": area_def.name, "relative_path": folder.relative_to(root).as_posix(), "created": True}

    def save_file(self, *, area: str, relative_path: str, metadata: dict[str, Any] | None, body: str, overwrite: bool = False) -> dict[str, Any]:
        area_def = self.knowledge._area(area)
        root = self.knowledge.root_dir / area_def.directory
        path = self._file_path(root, relative_path, must_exist=False)
        if path.exists() and not overwrite:
            raise ValueError("File already exists. Use overwrite=true to update it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".md":
            final_metadata = self._enforce_policy(area_def.name, metadata or self.metadata_template(area=area_def.name, relative_path=relative_path))
            text = self._compose_markdown(final_metadata, body or "")
        else:
            final_metadata = {}
            text = body or ""
        path.write_text(text, encoding="utf-8")
        validation = KnowledgeGovernanceService(knowledge=self.knowledge).validate_file(area_def.name, path.relative_to(root).as_posix()) if suffix == ".md" else {}
        return {
            "kind": "knowledge_editor_file_saved",
            "area": area_def.name,
            "relative_path": path.relative_to(root).as_posix(),
            "saved": True,
            "overwrite": overwrite,
            "governance": validation,
        }

    def move_file(self, *, source_area: str, source_path: str, target_area: str, target_path: str, overwrite: bool = False) -> dict[str, Any]:
        source_def = self.knowledge._area(source_area)
        target_def = self.knowledge._area(target_area)
        source_root = self.knowledge.root_dir / source_def.directory
        target_root = self.knowledge.root_dir / target_def.directory
        src = self._file_path(source_root, source_path, must_exist=True)
        dst = self._file_path(target_root, target_path, must_exist=False)
        if dst.exists() and not overwrite:
            raise ValueError("Target file already exists. Use overwrite=true to replace it.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8", errors="replace")
        if dst.suffix.lower() == ".md":
            metadata, body = parse_frontmatter(text)
            text = self._compose_markdown(self._enforce_policy(target_def.name, metadata or self.metadata_template(area=target_def.name, relative_path=target_path)), body)
            dst.write_text(text, encoding="utf-8")
            src.unlink()
        else:
            shutil.move(str(src), str(dst))
        return {"kind": "knowledge_editor_file_moved", "source_area": source_def.name, "source_path": source_path, "target_area": target_def.name, "target_path": dst.relative_to(target_root).as_posix(), "moved": True}

    def delete_path(self, *, area: str, relative_path: str, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            raise ValueError("Delete requires confirm=true.")
        area_def = self.knowledge._area(area)
        root = self.knowledge.root_dir / area_def.directory
        path = self._safe_path(root, relative_path)
        if not path.exists():
            raise FileNotFoundError("Knowledge path not found.")
        if path.is_dir():
            shutil.rmtree(path)
            deleted_type = "folder"
        else:
            path.unlink()
            deleted_type = "file"
        return {"kind": "knowledge_editor_deleted", "area": area_def.name, "relative_path": relative_path, "deleted_type": deleted_type, "deleted": True}

    def _file_path(self, root: Path, relative_path: str, *, must_exist: bool) -> Path:
        path = self._safe_path(root, relative_path)
        if path.suffix.lower() not in _ALLOWED_FILE_SUFFIXES:
            raise ValueError(f"Unsupported file type. Allowed: {sorted(_ALLOWED_FILE_SUFFIXES)}")
        if must_exist and (not path.exists() or not path.is_file()):
            raise FileNotFoundError("Knowledge file not found.")
        return path

    def _folder_path(self, root: Path, relative_path: str) -> Path:
        path = self._safe_path(root, relative_path)
        if path.suffix:
            raise ValueError("Folder path must not include a file suffix.")
        return path

    def _safe_path(self, root: Path, relative_path: str) -> Path:
        rel = (relative_path or "").strip().replace("\\", "/")
        if not rel or rel.startswith("/") or "\x00" in rel:
            raise ValueError("Invalid knowledge path.")
        if not _ALLOWED_NAME.match(rel):
            raise ValueError("Path contains unsupported characters.")
        root_resolved = root.resolve()
        candidate = (root / rel).resolve()
        if root_resolved != candidate and root_resolved not in candidate.parents:
            raise ValueError("Path escapes allowed knowledge area")
        return candidate

    def _enforce_policy(self, area: str, metadata: dict[str, Any]) -> dict[str, Any]:
        meta = dict(metadata or {})
        meta["visibility"] = area
        if area == "private_local_only":
            meta["cloud_allowed"] = False
        elif "cloud_allowed" not in meta:
            meta["cloud_allowed"] = True
        priority = str(meta.get("priority") or "normal").lower()
        if priority not in VALID_PRIORITIES:
            meta["priority"] = "normal"
        if meta.get("visibility") not in VALID_VISIBILITIES:
            meta["visibility"] = area
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [item.strip() for item in tags.split(",") if item.strip()]
        meta["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
        meta.setdefault("title", "Neue Notiz")
        meta.setdefault("owner", "thomas")
        meta.setdefault("last_reviewed", date.today().isoformat())
        meta.setdefault("summary", "")
        return meta

    def _compose_markdown(self, metadata: dict[str, Any], body: str) -> str:
        meta = dict(metadata)
        lines = ["---"]
        for key in ("title", "visibility", "cloud_allowed", "priority", "owner", "last_reviewed", "summary"):
            value = meta.get(key, "")
            if isinstance(value, bool):
                value = "true" if value else "false"
            lines.append(f"{key}: {value}")
        lines.append("tags:")
        tags = meta.get("tags") or []
        if tags:
            for tag in tags:
                lines.append(f"  - {tag}")
        lines.append("---")
        return "\n".join(lines) + "\n\n" + (body or "").lstrip()
