from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .obsidian_vault import ObsidianSafetyError, ObsidianVaultService, DEFAULT_INBOX_SUBDIRS

ALLOWED_REVIEW_STATES = {"pending", "reviewed", "accepted_for_sorting", "needs_revision", "rejected"}


class ObsidianInboxReviewService(ObsidianVaultService):
    """Review Pandora-owned notes inside the Obsidian Pandora_Inbox.

    This service intentionally only touches files below the configured inbox. It
    does not move files into the user's real vault structure and it never deletes
    notes. Updating review metadata is allowed because these notes were created
    by Pandora in the review inbox.
    """

    def status(self) -> dict[str, Any]:
        base = super().status()
        if not base.get("ok"):
            return {"kind": "obsidian_inbox_review_status", "ok": False, "base_status": base, "counts": {}}
        inbox = self._safe_inbox_path()
        inbox.mkdir(parents=True, exist_ok=True)
        for subdir in DEFAULT_INBOX_SUBDIRS:
            (inbox / subdir).mkdir(parents=True, exist_ok=True)
        items = self.list_items(limit=10000)["items"]
        counts: dict[str, int] = {}
        for item in items:
            status = item.get("review_status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return {
            "kind": "obsidian_inbox_review_status",
            "ok": True,
            "inbox": str(inbox),
            "item_count": len(items),
            "counts": counts,
        }

    def list_items(self, *, status: str | None = None, category: str | None = None, limit: int = 200) -> dict[str, Any]:
        inbox = self._safe_inbox_path()
        if not inbox.exists():
            return {"kind": "obsidian_inbox_items", "ok": True, "items": [], "item_count": 0}
        items: list[dict[str, Any]] = []
        for path in sorted(inbox.rglob("*.md")):
            self._assert_inside(path, inbox)
            rel = path.relative_to(inbox).as_posix()
            record = self._inbox_record(path, rel)
            if status and record.get("review_status") != status:
                continue
            if category and record.get("category") != category:
                continue
            items.append(record)
            if len(items) >= limit:
                break
        return {
            "kind": "obsidian_inbox_items",
            "ok": True,
            "item_count": len(items),
            "items": items,
        }

    def show_item(self, item_path: str) -> dict[str, Any]:
        path = self._safe_inbox_file(item_path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(self._safe_inbox_path()).as_posix()
        record = self._inbox_record(path, rel)
        record["content"] = text
        return {"kind": "obsidian_inbox_item", "ok": True, "item": record}

    def mark_item(self, item_path: str, *, status: str, note: str | None = None, reviewed_by: str = "user") -> dict[str, Any]:
        status = (status or "").strip()
        if status not in ALLOWED_REVIEW_STATES:
            raise ObsidianSafetyError(f"Unsupported review status: {status}")
        path = self._safe_inbox_file(item_path)
        original = path.read_text(encoding="utf-8", errors="ignore")
        metadata, body, had_frontmatter = self._split_frontmatter(original)
        metadata["review_status"] = status
        metadata["reviewed_by"] = reviewed_by
        metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        if note:
            metadata["review_note"] = note.replace("\n", " ").strip()
        updated = self._render_frontmatter(metadata) + "\n" + body.lstrip("\n")
        path.write_text(updated, encoding="utf-8")
        rel = path.relative_to(self._safe_inbox_path()).as_posix()
        return {
            "kind": "obsidian_inbox_mark",
            "ok": True,
            "relative_path": rel,
            "review_status": status,
            "had_frontmatter": had_frontmatter,
            "write_policy": "metadata_update_in_inbox_only",
        }

    def _safe_inbox_file(self, item_path: str) -> Path:
        inbox = self._safe_inbox_path()
        cleaned = (item_path or "").strip().replace("\\", "/").lstrip("/")
        if not cleaned or ".." in Path(cleaned).parts:
            raise ObsidianSafetyError("Unsafe Obsidian inbox item path")
        target = (inbox / cleaned).resolve()
        self._assert_inside(target, inbox)
        if target.suffix.lower() != ".md":
            raise ObsidianSafetyError("Only Markdown inbox items are supported")
        if not target.exists() or not target.is_file():
            raise ObsidianSafetyError("Obsidian inbox item does not exist")
        return target

    def _inbox_record(self, path: Path, rel: str) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata, body, _ = self._split_frontmatter(text)
        category = rel.split("/", 1)[0] if "/" in rel else "Inbox"
        title = str(metadata.get("title") or self._extract_title(body) or Path(rel).stem)
        return {
            "relative_path": rel,
            "title": title,
            "category": category,
            "review_status": str(metadata.get("review_status") or "pending"),
            "suggested_folder": str(metadata.get("suggested_folder") or ""),
            "generated_at": str(metadata.get("generated_at") or ""),
            "generated_by": str(metadata.get("generated_by") or ""),
            "tags": self._metadata_list(metadata.get("tags")),
            "word_count": len(re.sub(r"\s+", " ", body).strip().split()) if body.strip() else 0,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "excerpt": re.sub(r"\s+", " ", body).strip()[:320],
        }

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str, bool]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text, False
        end_index = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_index = idx
                break
        if end_index is None:
            return {}, text, False
        meta_lines = lines[1:end_index]
        body = "\n".join(lines[end_index + 1:]) + ("\n" if text.endswith("\n") else "")
        return self._parse_simple_yaml(meta_lines), body, True

    def _parse_simple_yaml(self, lines: list[str]) -> dict[str, Any]:
        data: dict[str, Any] = {}
        current_list_key: str | None = None
        for raw in lines:
            line = raw.rstrip()
            if not line.strip():
                continue
            if line.startswith("  - ") and current_list_key:
                data.setdefault(current_list_key, []).append(line[4:].strip().strip('"\''))
                continue
            current_list_key = None
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = value.strip().strip('"\'')
        return data

    def _render_frontmatter(self, metadata: dict[str, Any]) -> str:
        preferred = ["title", "generated_by", "generated_at", "review_status", "reviewed_by", "reviewed_at", "review_note", "cloud_allowed", "suggested_folder", "tags"]
        lines = ["---"]
        emitted = set()
        for key in preferred + sorted(k for k in metadata if k not in preferred):
            if key in emitted or key not in metadata:
                continue
            emitted.add(key)
            value = metadata[key]
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("---")
        return "\n".join(lines)

    def _metadata_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str) and value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return []
