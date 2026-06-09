from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end].splitlines()
    body_start = text.find("\n", end + 1) + 1
    body = text[body_start:] if body_start > 0 else ""
    meta: dict[str, Any] = {}
    current_key: str | None = None
    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and current_key:
            meta.setdefault(current_key, [])
            if isinstance(meta[current_key], list):
                meta[current_key].append(_scalar(stripped[1:].strip()))
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            meta[current_key] = [] if value == "" else _scalar(value)
    return meta, body


def _scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    lower = value.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    return value


def normalize_metadata(area: str, relative_path: str, text: str) -> dict[str, Any]:
    raw, body = parse_frontmatter(text)
    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [item.strip() for item in tags.split(",") if item.strip()]
    visibility = str(raw.get("visibility") or area).strip()
    priority = str(raw.get("priority") or "normal").strip().lower()
    cloud_allowed = raw.get("cloud_allowed")
    if cloud_allowed is None:
        cloud_allowed = area != "private_local_only"
    return {
        "title": str(raw.get("title") or Path(relative_path).stem),
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        "visibility": visibility,
        "cloud_allowed": bool(cloud_allowed),
        "priority": priority,
        "owner": str(raw.get("owner") or ""),
        "last_reviewed": str(raw.get("last_reviewed") or ""),
        "summary": str(raw.get("summary") or ""),
        "has_frontmatter": bool(raw),
        "content_without_frontmatter_chars": len(body),
    }


def strip_frontmatter(text: str) -> str:
    _, body = parse_frontmatter(text)
    return body
