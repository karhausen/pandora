from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MEMORY_DIR, PROPOSALS_DIR


@dataclass
class MemoryExplorerService:
    """Read-only GUI/API service for Pandora memory files.

    The explorer deliberately does not mutate memory. It gives the user a safe
    overview, light-weight search and bounded previews for JSON/JSONL/TXT/MD
    artifacts in the allowed memory-related folders.
    """

    memory_dir: Path = MEMORY_DIR
    proposals_dir: Path = PROPOSALS_DIR
    max_preview_chars: int = 6000

    def dashboard(self, *, query: str | None = None, limit: int = 20) -> dict[str, Any]:
        areas = self.areas()
        results = self.search(query=query or "", limit=limit) if query else {"query": "", "count": 0, "results": []}
        return {
            "kind": "memory_explorer_dashboard",
            "area_count": len(areas["areas"]),
            "total_files": sum(area["file_count"] for area in areas["areas"]),
            "total_bytes": sum(area["total_bytes"] for area in areas["areas"]),
            "areas": areas["areas"],
            "search": results,
            "read_only": True,
        }

    def areas(self) -> dict[str, Any]:
        area_defs = [
            ("memory", self.memory_dir, "Core-, Chat-, Task-, Tool- und Lernspeicher"),
            ("proposals", self.proposals_dir, "Review-, Improvement- und Candidate-Vorschläge"),
        ]
        areas: list[dict[str, Any]] = []
        for name, path, description in area_defs:
            files = self._scan_files(path)
            areas.append(
                {
                    "name": name,
                    "path": str(path),
                    "description": description,
                    "exists": path.exists(),
                    "file_count": len(files),
                    "total_bytes": sum(file["size_bytes"] for file in files),
                    "recent_files": files[:10],
                }
            )
        return {"areas": areas}

    def list_area(self, area: str, *, limit: int = 200) -> dict[str, Any]:
        root = self._area_root(area)
        files = self._scan_files(root)[:limit]
        return {"area": area, "path": str(root), "count": len(files), "files": files, "read_only": True}

    def show_file(self, area: str, relative_path: str, *, max_lines: int = 120) -> dict[str, Any]:
        root = self._area_root(area)
        file_path = self._safe_path(root, relative_path)
        if not file_path.exists() or not file_path.is_file():
            return {"found": False, "area": area, "relative_path": relative_path, "error": "Memory file not found"}
        suffix = file_path.suffix.lower()
        stat = file_path.stat()
        payload: dict[str, Any] = {
            "found": True,
            "area": area,
            "relative_path": file_path.relative_to(root).as_posix(),
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "type": suffix.lstrip(".") or "file",
            "read_only": True,
        }
        if suffix == ".json":
            payload["content"] = self._load_json(file_path)
            payload["preview"] = json.dumps(payload["content"], indent=2, ensure_ascii=False)[: self.max_preview_chars]
        elif suffix == ".jsonl":
            lines = self._load_jsonl(file_path, max_lines=max_lines)
            payload["entries"] = lines
            payload["entry_count_preview"] = len(lines)
            payload["preview"] = json.dumps(lines, indent=2, ensure_ascii=False)[: self.max_preview_chars]
        else:
            text = self._safe_read_text(file_path)
            payload["preview"] = "\n".join(text.splitlines()[:max_lines])[: self.max_preview_chars]
        return payload

    def search(self, *, query: str, limit: int = 50) -> dict[str, Any]:
        needle = (query or "").strip().lower()
        if not needle:
            return {"query": query, "count": 0, "results": []}
        results: list[dict[str, Any]] = []
        for area in ("memory", "proposals"):
            root = self._area_root(area)
            for file in self._scan_files(root, limit=500):
                path = root / file["relative_path"]
                haystack = f"{file['relative_path']}\n{self._safe_read_text(path)[:20000]}".lower()
                if needle in haystack:
                    results.append({"area": area, **file, "snippet": self._snippet(self._safe_read_text(path), needle)})
                    if len(results) >= limit:
                        return {"query": query, "count": len(results), "results": results, "truncated": True}
        return {"query": query, "count": len(results), "results": results, "truncated": False}

    def _area_root(self, area: str) -> Path:
        normalized = area.strip().lower()
        if normalized == "memory":
            return self.memory_dir
        if normalized == "proposals":
            return self.proposals_dir
        raise ValueError("Unsupported memory area. Allowed: memory, proposals")

    def _safe_path(self, root: Path, relative_path: str) -> Path:
        root_resolved = root.resolve()
        candidate = (root / relative_path).resolve()
        if root_resolved != candidate and root_resolved not in candidate.parents:
            raise ValueError("Path escapes allowed memory area")
        return candidate

    def _scan_files(self, root: Path, limit: int | None = None) -> list[dict[str, Any]]:
        if not root.exists():
            return []
        files: list[dict[str, Any]] = []
        allowed = {".json", ".jsonl", ".txt", ".md"}
        for path in root.rglob("*"):
            if not path.is_file() or path.name == ".gitkeep" or path.suffix.lower() not in allowed:
                continue
            stat = path.stat()
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "name": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )
        files.sort(key=lambda item: item["modified_at"], reverse=True)
        return files[:limit] if limit else files

    def _load_json(self, path: Path) -> Any:
        try:
            return json.loads(self._safe_read_text(path) or "{}")
        except json.JSONDecodeError as exc:
            return {"_error": f"Invalid JSON: {exc}"}

    def _load_jsonl(self, path: Path, *, max_lines: int) -> list[Any]:
        lines = self._safe_read_text(path).splitlines()[-max_lines:]
        entries: list[Any] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({"_raw": line, "_warning": "Invalid JSONL line"})
        return entries

    def _safe_read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<read error: {exc}>"

    def _snippet(self, text: str, needle: str, width: int = 220) -> str:
        lower = text.lower()
        idx = lower.find(needle)
        if idx < 0:
            return text[:width]
        start = max(0, idx - width // 3)
        end = min(len(text), idx + width)
        return text[start:end].replace("\n", " ").strip()
