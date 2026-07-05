from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .knowledge_metadata import normalize_metadata, strip_frontmatter

from .config import ROOT_DIR


@dataclass(frozen=True)
class KnowledgeArea:
    name: str
    directory: str
    policy: str
    cloud_allowed: bool
    description: str


AREAS: tuple[KnowledgeArea, ...] = (
    KnowledgeArea(
        name="public",
        directory="public",
        policy="public_cloud_allowed",
        cloud_allowed=True,
        description="Allgemeines Wissen, das lokal und in Cloud-LLMs verwendet werden darf.",
    ),
    KnowledgeArea(
        name="restricted_cloud_allowed",
        directory="restricted_cloud_allowed",
        policy="restricted_cloud_allowed",
        cloud_allowed=True,
        description="Technische oder organisatorische Notizen, die nach Policy-Prüfung in Cloud-Kontext dürfen.",
    ),
    KnowledgeArea(
        name="private_local_only",
        directory="private_local_only",
        policy="local_only",
        cloud_allowed=False,
        description="Private oder firmennahe Notizen. Niemals an Cloud-LLMs geben.",
    ),
)


@dataclass
class UserKnowledgeBaseService:
    """Read-only user knowledge base for markdown/text/json notes.

    This service separates user-provided knowledge from Pandora runtime memory.
    It deliberately exposes an explicit cloud policy per branch so later context
    injection can decide whether a file is allowed for local-only or cloud LLMs.
    """

    root_dir: Path = ROOT_DIR / "user_knowledge"
    max_preview_chars: int = 8000
    allowed_suffixes: tuple[str, ...] = (".md", ".txt", ".json")

    def ensure_structure(self) -> dict[str, Any]:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for area in AREAS:
            area_root = self.root_dir / area.directory
            area_root.mkdir(parents=True, exist_ok=True)
            (area_root / ".gitkeep").touch(exist_ok=True)
        readme = self.root_dir / "README.md"
        if not readme.exists():
            readme.write_text(self._default_readme(), encoding="utf-8")
        return self.status()

    def status(self) -> dict[str, Any]:
        areas = self.areas()["areas"]
        return {
            "kind": "user_knowledge_base_status",
            "root": str(self.root_dir),
            "exists": self.root_dir.exists(),
            "area_count": len(areas),
            "total_files": sum(area["file_count"] for area in areas),
            "total_bytes": sum(area["total_bytes"] for area in areas),
            "cloud_allowed_files": sum(area["file_count"] for area in areas if area["cloud_allowed"]),
            "local_only_files": sum(area["file_count"] for area in areas if not area["cloud_allowed"]),
            "read_only": True,
        }

    def dashboard(self, *, query: str | None = None, limit: int = 20) -> dict[str, Any]:
        self.ensure_structure()
        areas = self.areas()["areas"]
        results = self.search(query=query or "", limit=limit) if query else {"query": "", "count": 0, "results": []}
        return {
            "kind": "user_knowledge_base_dashboard",
            "root": str(self.root_dir),
            "areas": areas,
            "area_count": len(areas),
            "total_files": sum(area["file_count"] for area in areas),
            "total_bytes": sum(area["total_bytes"] for area in areas),
            "search": results,
            "read_only": True,
            "policy_summary": {
                "public": "Cloud erlaubt",
                "restricted_cloud_allowed": "Cloud erlaubt nach Policy-Prüfung",
                "private_local_only": "Nur lokales LLM, niemals Cloud",
            },
        }

    def areas(self) -> dict[str, Any]:
        payload: list[dict[str, Any]] = []
        for area in AREAS:
            root = self.root_dir / area.directory
            files = self._scan_files(root)
            payload.append(
                {
                    "name": area.name,
                    "directory": area.directory,
                    "path": str(root),
                    "policy": area.policy,
                    "cloud_allowed": area.cloud_allowed,
                    "description": area.description,
                    "exists": root.exists(),
                    "file_count": len(files),
                    "total_bytes": sum(file["size_bytes"] for file in files),
                    "recent_files": files[:10],
                }
            )
        return {"areas": payload}

    def list_area(self, area: str, *, limit: int = 200) -> dict[str, Any]:
        area_def = self._area(area)
        root = self.root_dir / area_def.directory
        files = self._scan_files(root)[:limit]
        return {
            "area": area_def.name,
            "path": str(root),
            "policy": area_def.policy,
            "cloud_allowed": area_def.cloud_allowed,
            "description": area_def.description,
            "count": len(files),
            "files": files,
            "read_only": True,
        }

    def show_file(self, area: str, relative_path: str, *, max_lines: int = 160) -> dict[str, Any]:
        area_def = self._area(area)
        root = self.root_dir / area_def.directory
        file_path = self._safe_path(root, relative_path)
        if not file_path.exists() or not file_path.is_file():
            return {"found": False, "area": area_def.name, "relative_path": relative_path, "error": "knowledge file not found"}
        if file_path.suffix.lower() not in self.allowed_suffixes:
            return {"found": False, "area": area_def.name, "relative_path": relative_path, "error": "unsupported file type"}
        stat = file_path.stat()
        text = self._safe_read_text(file_path)
        metadata = normalize_metadata(area_def.name, file_path.relative_to(root).as_posix(), text) if file_path.suffix.lower() == ".md" else {}
        preview_text = strip_frontmatter(text) if file_path.suffix.lower() == ".md" else text
        preview = "\n".join(preview_text.splitlines()[:max_lines])[: self.max_preview_chars]
        payload: dict[str, Any] = {
            "found": True,
            "area": area_def.name,
            "relative_path": file_path.relative_to(root).as_posix(),
            "name": file_path.name,
            "type": file_path.suffix.lower().lstrip("."),
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "policy": area_def.policy,
            "cloud_allowed": bool(metadata.get("cloud_allowed", area_def.cloud_allowed)),
            "local_only": not bool(metadata.get("cloud_allowed", area_def.cloud_allowed)),
            "metadata": metadata,
            "tags": metadata.get("tags", []),
            "priority": metadata.get("priority", "normal"),
            "title": metadata.get("title") or file_path.stem,
            "preview": preview,
            "read_only": True,
        }
        if file_path.suffix.lower() == ".json":
            payload["content"] = self._load_json(file_path)
        return payload

    def search(self, *, query: str, limit: int = 50, cloud_context: bool = False) -> dict[str, Any]:
        self.ensure_structure()
        needle = (query or "").strip().lower()
        if not needle:
            return {"query": query, "count": 0, "results": [], "cloud_context": cloud_context}
        results: list[dict[str, Any]] = []
        for area_def in AREAS:
            if cloud_context and not area_def.cloud_allowed:
                continue
            root = self.root_dir / area_def.directory
            for file in self._scan_files(root, limit=1000):
                path = root / file["relative_path"]
                text = self._safe_read_text(path)
                metadata = file.get("metadata", {}) or {}
                if cloud_context and not bool(metadata.get("cloud_allowed", area_def.cloud_allowed)):
                    continue
                meta_text = " ".join([str(metadata.get("title", "")), str(metadata.get("summary", "")), " ".join(metadata.get("tags", []) or [])])
                haystack = f"{file['relative_path']}\n{meta_text}\n{text[:50000]}".lower()
                if needle in haystack:
                    score = self._score_match(needle, file, metadata, text)
                    results.append(
                        {
                            "area": area_def.name,
                            "policy": area_def.policy,
                            "cloud_allowed": bool(metadata.get("cloud_allowed", area_def.cloud_allowed)),
                            **file,
                            "metadata": metadata,
                            "score": score,
                            "snippet": self._snippet(text, needle),
                        }
                    )
        results.sort(key=lambda item: (item.get("score", 0), item.get("modified_at", 0)), reverse=True)
        truncated = len(results) > limit
        return {"query": query, "count": min(len(results), limit), "results": results[:limit], "cloud_context": cloud_context, "truncated": truncated}

    def context_preview(self, *, query: str, target: str = "local", limit: int = 10) -> dict[str, Any]:
        """Return a safe preview of knowledge files eligible for a target LLM context."""
        normalized = (target or "local").strip().lower()
        cloud_context = normalized in {"cloud", "cloud_llm", "company", "company_llm"}
        results = self.search(query=query, limit=limit, cloud_context=cloud_context)
        blocked_local_only = 0
        if cloud_context:
            full = self.search(query=query, limit=500, cloud_context=False)
            blocked_local_only = sum(1 for item in full["results"] if not item.get("cloud_allowed"))
        return {
            "kind": "user_knowledge_context_preview",
            "target": normalized,
            "cloud_context": cloud_context,
            "query": query,
            "allowed_count": results["count"],
            "blocked_local_only_count": blocked_local_only,
            "results": results["results"],
            "rule": "private_local_only is never included for cloud/company targets",
        }

    def _area(self, area: str) -> KnowledgeArea:
        normalized = (area or "").strip().lower()
        for item in AREAS:
            if normalized in {item.name, item.directory}:
                return item
        allowed = ", ".join(item.name for item in AREAS)
        raise ValueError(f"Unsupported knowledge area. Allowed: {allowed}")

    def _safe_path(self, root: Path, relative_path: str) -> Path:
        root_resolved = root.resolve()
        candidate = (root / relative_path).resolve()
        if root_resolved != candidate and root_resolved not in candidate.parents:
            raise ValueError("Path escapes allowed knowledge area")
        return candidate

    def _scan_files(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        if not root.exists():
            return []
        files: list[dict[str, Any]] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.name == ".gitkeep" or path.suffix.lower() not in self.allowed_suffixes:
                continue
            stat = path.stat()
            relative_path = path.relative_to(root).as_posix()
            area_name = root.name
            metadata = normalize_metadata(area_name, relative_path, self._safe_read_text(path)) if path.suffix.lower() == ".md" else {}
            files.append(
                {
                    "relative_path": relative_path,
                    "name": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "metadata": metadata,
                    "tags": metadata.get("tags", []),
                    "priority": metadata.get("priority", "normal"),
                    "title": metadata.get("title") or path.stem,
                }
            )
        files.sort(key=lambda item: item["modified_at"], reverse=True)
        return files[:limit] if limit else files

    def _safe_read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<read error: {exc}>"

    def _load_json(self, path: Path) -> Any:
        try:
            return json.loads(self._safe_read_text(path) or "{}")
        except json.JSONDecodeError as exc:
            return {"_error": f"Invalid JSON: {exc}"}

    def _score_match(self, needle: str, file: dict[str, Any], metadata: dict[str, Any], text: str) -> int:
        score = 0
        if needle in str(metadata.get("title", "")).lower():
            score += 30
        if any(needle in str(tag).lower() for tag in metadata.get("tags", []) or []):
            score += 25
        if needle in file.get("relative_path", "").lower():
            score += 15
        if needle in text.lower():
            score += 10
        priority = metadata.get("priority") or "normal"
        score += {"critical": 12, "high": 8, "normal": 3, "low": 0}.get(priority, 1)
        return score

    def _snippet(self, text: str, needle: str, width: int = 260) -> str:
        lower = text.lower()
        idx = lower.find(needle)
        if idx < 0:
            return text[:width]
        start = max(0, idx - width // 3)
        end = min(len(text), idx + width)
        return text[start:end].replace("\n", " ").strip()

    def _default_readme(self) -> str:
        return """# Pandora User Knowledge Base\n\nHier kannst du eigene Markdown-, Text- oder JSON-Notizen ablegen, die Pandora als Wissensbasis verwenden darf.\n\n## Bereiche\n\n- `public/`: darf lokal und in Cloud-LLMs verwendet werden.\n- `restricted_cloud_allowed/`: darf erst nach Policy-Prüfung in Cloud-Kontext.\n- `private_local_only/`: nur lokales LLM, niemals Cloud.\n\nEmpfehlung: Themen als Unterordner ablegen, z. B. `public/python/`, `public/funkgeraete/` oder `private_local_only/firma/`.\n"""
