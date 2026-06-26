from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INBOX_SUBDIRS = ("Knowledge", "Skills", "Research", "Drafts")


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class ObsidianConfig:
    enabled: bool
    vault_path: Path | None
    inbox_dir: str
    mode: str
    cloud_allowed: bool
    company_allowed: bool

    @classmethod
    def load(cls, root_dir: Path = ROOT) -> "ObsidianConfig":
        env_file = _read_env_file(root_dir / ".env")

        def get(name: str, default: str | None = None) -> str | None:
            return os.environ.get(name, env_file.get(name, default))

        raw_path = get("OBSIDIAN_VAULT_PATH", "") or ""
        vault_path = Path(raw_path).expanduser() if raw_path.strip() else None
        inbox_dir = (get("OBSIDIAN_INBOX_DIR", "Pandora_Inbox") or "Pandora_Inbox").strip().strip("/\\")
        return cls(
            enabled=_env_bool(get("OBSIDIAN_VAULT_ENABLED", "false")),
            vault_path=vault_path,
            inbox_dir=inbox_dir or "Pandora_Inbox",
            mode=get("OBSIDIAN_MODE", "read_write_inbox_only") or "read_write_inbox_only",
            cloud_allowed=_env_bool(get("OBSIDIAN_CLOUD_ALLOWED", "false")),
            company_allowed=_env_bool(get("OBSIDIAN_COMPANY_ALLOWED", "false")),
        )

    def public_dict(self) -> dict[str, Any]:
        path = str(self.vault_path) if self.vault_path else None
        return {
            "enabled": self.enabled,
            "vault_path_configured": bool(self.vault_path),
            "vault_path": path,
            "inbox_dir": self.inbox_dir,
            "mode": self.mode,
            "cloud_allowed": self.cloud_allowed,
            "company_allowed": self.company_allowed,
        }


class ObsidianSafetyError(ValueError):
    pass


class ObsidianVaultService:
    """Read an Obsidian vault and write only into Pandora_Inbox.

    This connector intentionally avoids deletes, moves and overwrites. It is a
    controlled bridge from Pandora into Obsidian, not a general file manager.
    """

    def __init__(self, root_dir: Path = ROOT, config: ObsidianConfig | None = None):
        self.root_dir = root_dir
        self.config = config or ObsidianConfig.load(root_dir)

    @property
    def vault_path(self) -> Path | None:
        return self.config.vault_path

    @property
    def inbox_path(self) -> Path | None:
        if not self.config.vault_path:
            return None
        return self.config.vault_path / self.config.inbox_dir

    def status(self) -> dict[str, Any]:
        vault = self.vault_path
        exists = bool(vault and vault.exists() and vault.is_dir())
        inbox = self.inbox_path
        inbox_exists = bool(inbox and inbox.exists() and inbox.is_dir())
        return {
            "kind": "obsidian_status",
            "config": self.config.public_dict(),
            "vault_exists": exists,
            "vault_readable": bool(exists and os.access(vault, os.R_OK)),
            "inbox_exists": inbox_exists,
            "inbox_writable": bool(inbox_exists and os.access(inbox, os.W_OK)),
            "write_policy": {
                "read_vault": True,
                "write_inbox_only": self.config.mode == "read_write_inbox_only",
                "delete_allowed": False,
                "move_allowed": False,
                "overwrite_allowed": False,
            },
            "ok": bool(self.config.enabled and exists),
            "issues": self._status_issues(),
        }

    def _status_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.config.enabled:
            issues.append("OBSIDIAN_VAULT_ENABLED is false")
        if not self.vault_path:
            issues.append("OBSIDIAN_VAULT_PATH is not configured")
        elif not self.vault_path.exists():
            issues.append("Configured Obsidian vault path does not exist")
        elif not self.vault_path.is_dir():
            issues.append("Configured Obsidian vault path is not a directory")
        if self.config.mode != "read_write_inbox_only":
            issues.append("Only OBSIDIAN_MODE=read_write_inbox_only is supported")
        return issues

    def ensure_inbox(self) -> dict[str, Any]:
        self._require_vault()
        inbox = self._safe_inbox_path()
        created: list[str] = []
        inbox.mkdir(parents=True, exist_ok=True)
        for subdir in DEFAULT_INBOX_SUBDIRS:
            path = inbox / subdir
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append((inbox.name + "/" + subdir))
        return {"ok": True, "inbox": str(inbox), "created": created}

    def index(self, *, limit: int = 10000, write: bool = True) -> dict[str, Any]:
        vault = self._require_vault()
        files = []
        for path in sorted(vault.rglob("*.md")):
            if self._is_ignored(path):
                continue
            rel = path.relative_to(vault).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            files.append(self._file_record(path, rel, text))
            if len(files) >= limit:
                break
        tags: dict[str, int] = {}
        links: dict[str, int] = {}
        for item in files:
            for tag in item["tags"]:
                tags[tag] = tags.get(tag, 0) + 1
            for link in item["wikilinks"]:
                links[link] = links.get(link, 0) + 1
        report = {
            "kind": "obsidian_index",
            "ok": True,
            "vault": str(vault),
            "file_count": len(files),
            "tag_count": len(tags),
            "wikilink_count": len(links),
            "cloud_allowed": self.config.cloud_allowed,
            "company_allowed": self.config.company_allowed,
            "files": files,
            "top_tags": sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))[:50],
            "top_wikilinks": sorted(links.items(), key=lambda kv: (-kv[1], kv[0]))[:50],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if write:
            data_dir = self.root_dir / "data" / "obsidian"
            data_dir.mkdir(parents=True, exist_ok=True)
            import json
            (data_dir / "index.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def search(self, query: str, *, limit: int = 20, include_content: bool = False) -> dict[str, Any]:
        q = (query or "").strip().lower()
        if not q:
            return {"kind": "obsidian_search", "ok": False, "query": query, "results": [], "error": "query is empty"}
        records = self.index(limit=10000, write=False)["files"]
        terms = [part for part in re.split(r"\s+", q) if part]
        results: list[dict[str, Any]] = []
        for rec in records:
            hay = " ".join([rec["title"], rec["relative_path"], " ".join(rec["tags"]), " ".join(rec["wikilinks"]), rec.get("excerpt", "")]).lower()
            score = sum(3 if term in rec["title"].lower() else 1 for term in terms if term in hay)
            if score <= 0:
                continue
            item = {k: rec[k] for k in ["relative_path", "title", "tags", "wikilinks", "word_count", "modified_at", "sha256", "metadata", "company_allowed", "cloud_allowed"]}
            item["score"] = score
            item["excerpt"] = rec.get("excerpt", "")
            if include_content:
                path = self._safe_vault_file(rec["relative_path"])
                item["content"] = path.read_text(encoding="utf-8", errors="ignore")
            results.append(item)
        results.sort(key=lambda item: (-item["score"], item["relative_path"]))
        return {
            "kind": "obsidian_search",
            "ok": True,
            "query": query,
            "result_count": len(results[:limit]),
            "cloud_allowed": self.config.cloud_allowed,
            "company_allowed": self.config.company_allowed,
            "results": results[:limit],
        }

    def tags(self, *, limit: int = 200) -> dict[str, Any]:
        idx = self.index(limit=10000, write=False)
        return {"kind": "obsidian_tags", "ok": True, "tags": idx["top_tags"][:limit], "tag_count": idx["tag_count"]}

    def export_markdown(
        self,
        *,
        title: str,
        content: str,
        category: str = "Knowledge",
        tags: list[str] | None = None,
        suggested_folder: str | None = None,
        source: str = "pandora",
    ) -> dict[str, Any]:
        self._require_vault()
        if self.config.mode != "read_write_inbox_only":
            raise ObsidianSafetyError("Only read_write_inbox_only export mode is supported")
        inbox = self._safe_inbox_path()
        category = self._safe_segment(category or "Knowledge")
        target_dir = inbox / category
        self._assert_inside(target_dir, inbox)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_title = self._safe_filename(title or "pandora_note")
        target = target_dir / f"{safe_title}.md"
        if target.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = target_dir / f"{safe_title}_{suffix}.md"
        self._assert_inside(target, inbox)
        tag_list = [self._clean_tag(tag) for tag in (tags or []) if self._clean_tag(tag)]
        frontmatter = [
            "---",
            f"title: {title}",
            f"generated_by: {source}",
            f"generated_at: {datetime.now(timezone.utc).isoformat()}",
            "review_status: pending",
            "cloud_allowed: false",
            "company_allowed: false",
            f"suggested_folder: {suggested_folder or ''}",
            "tags:",
        ]
        frontmatter.extend([f"  - {tag}" for tag in tag_list] or ["  - pandora"])
        frontmatter.append("---")
        markdown = "\n".join(frontmatter).strip() + "\n\n" + (content or "").strip() + "\n"
        target.write_text(markdown, encoding="utf-8")
        rel = target.relative_to(self._require_vault()).as_posix()
        return {
            "kind": "obsidian_export",
            "ok": True,
            "relative_path": rel,
            "path": str(target),
            "category": category,
            "write_policy": "inbox_only_no_overwrite",
        }

    def _file_record(self, path: Path, rel: str, text: str) -> dict[str, Any]:
        metadata = self._extract_frontmatter(text)
        title = str(metadata.get("title") or self._extract_title(text) or Path(rel).stem)
        tags = sorted(set(re.findall(r"(?<!\w)#([A-Za-z0-9_/-]+)", text)) | set(self._metadata_tags(metadata)))
        wikilinks = sorted(set(match.strip() for match in re.findall(r"\[\[([^\]]+)\]\]", text)))
        cleaned = re.sub(r"\s+", " ", self._strip_frontmatter(text)).strip()
        return {
            "relative_path": rel,
            "title": title,
            "tags": tags,
            "wikilinks": wikilinks,
            "metadata": metadata,
            "company_allowed": self._metadata_bool(metadata.get("company_allowed"), self.config.company_allowed),
            "cloud_allowed": self._metadata_bool(metadata.get("cloud_allowed"), self.config.cloud_allowed),
            "word_count": len(cleaned.split()) if cleaned else 0,
            "excerpt": cleaned[:320],
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
        }

    def _strip_frontmatter(self, text: str) -> str:
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                return parts[2]
        return text

    def _extract_frontmatter(self, text: str) -> dict[str, Any]:
        if not text.startswith("---"):
            return {}
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}
        raw = parts[1]
        data: dict[str, Any] = {}
        current_key: str | None = None
        for line in raw.splitlines():
            if not line.strip():
                continue
            if line.startswith("  -") and current_key:
                data.setdefault(current_key, []).append(line.split("-", 1)[1].strip())
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value == "":
                data[key] = [] if key == "tags" else ""
                current_key = key
            elif value.lower() in {"true", "false"}:
                data[key] = value.lower() == "true"
                current_key = key
            else:
                data[key] = value
                current_key = key
        return data

    def _metadata_tags(self, metadata: dict[str, Any]) -> list[str]:
        value = metadata.get("tags", [])
        if isinstance(value, list):
            return [str(v).strip().lstrip("#") for v in value if str(v).strip()]
        if isinstance(value, str):
            return [part.strip().lstrip("#") for part in value.split(",") if part.strip()]
        return []

    def _metadata_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or value == "":
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}

    def _extract_title(self, text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            if stripped.lower().startswith("title:"):
                return stripped.split(":", 1)[1].strip().strip('"')
        return None

    def _is_ignored(self, path: Path) -> bool:
        parts = set(path.parts)
        return ".obsidian" in parts or ".trash" in parts or path.name.startswith(".")

    def _require_vault(self) -> Path:
        vault = self.vault_path
        if not self.config.enabled:
            raise ObsidianSafetyError("Obsidian vault integration is disabled")
        if not vault:
            raise ObsidianSafetyError("OBSIDIAN_VAULT_PATH is not configured")
        if not vault.exists() or not vault.is_dir():
            raise ObsidianSafetyError("Configured Obsidian vault path is not available")
        return vault.resolve()

    def _safe_inbox_path(self) -> Path:
        vault = self._require_vault()
        inbox = (vault / self.config.inbox_dir).resolve()
        self._assert_inside(inbox, vault)
        if inbox == vault:
            raise ObsidianSafetyError("Inbox directory may not be the vault root")
        return inbox

    def _safe_vault_file(self, relative_path: str) -> Path:
        vault = self._require_vault()
        target = (vault / relative_path).resolve()
        self._assert_inside(target, vault)
        if target.suffix.lower() != ".md":
            raise ObsidianSafetyError("Only Markdown files are supported")
        return target

    def _assert_inside(self, target: Path, base: Path) -> None:
        try:
            target.resolve().relative_to(base.resolve())
        except ValueError as exc:
            raise ObsidianSafetyError("Path escapes the configured Obsidian vault/inbox") from exc

    def _safe_segment(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß _.-]", "_", value).strip().strip("./\\")
        if not cleaned or cleaned in {".", ".."} or ".." in cleaned:
            raise ObsidianSafetyError("Unsafe Obsidian path segment")
        return cleaned

    def _safe_filename(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß _.-]", "_", value).strip().strip(".")
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned[:90] or "pandora_note"

    def _clean_tag(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_/-]", "", value.strip().lstrip("#"))[:80]
