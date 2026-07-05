from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .knowledge_metadata import normalize_metadata, strip_frontmatter
from .user_knowledge_base import AREAS, UserKnowledgeBaseService

VALID_VISIBILITIES = {"public", "restricted_cloud_allowed", "private_local_only"}
VALID_PRIORITIES = {"low", "normal", "high", "critical"}
REVIEW_MAX_AGE_DAYS = 180
MIN_TAGS = 1
MIN_MEANINGFUL_WORDS = 8
MAX_CONTEXT_CHARS_RECOMMENDED = 60_000

REQUIRED_FRONTMATTER_FIELDS = (
    "title",
    "tags",
    "visibility",
    "cloud_allowed",
    "priority",
    "last_reviewed",
)


@dataclass
class KnowledgeGovernanceService:
    """Read-only governance/audit service for user-managed knowledge files.

    This service performs real policy checks against the file's folder, parsed
    markdown frontmatter and content shape. It never moves, edits or deletes user
    files. All results are meant for review in CLI/API/GUI.
    """

    knowledge: UserKnowledgeBaseService = field(default_factory=UserKnowledgeBaseService)

    def status(self) -> dict[str, Any]:
        report = self.run(limit=500)
        return {
            "kind": "knowledge_governance_status",
            "ok": report["error_count"] == 0,
            "file_count": report["file_count"],
            "error_count": report["error_count"],
            "warning_count": report["warning_count"],
            "info_count": report["info_count"],
            "review_recommended_count": report["review_recommended_count"],
            "health_score": report["health_score"],
            "grade": report["grade"],
        }

    def metadata_index(self, *, limit: int = 500) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for file_ref in self._iter_file_refs(limit=limit):
            shown = self.knowledge.show_file(file_ref["area"], file_ref["relative_path"], max_lines=500)
            validation = self.validate_file(file_ref["area"], file_ref["relative_path"])
            files.append(
                {
                    "area": file_ref["area"],
                    "relative_path": file_ref["relative_path"],
                    "name": file_ref.get("name"),
                    "policy": file_ref["policy"],
                    "metadata": shown.get("metadata", {}),
                    "governance": validation,
                }
            )
        return {"kind": "knowledge_metadata_index", "count": len(files), "files": files, "truncated": len(files) >= limit}

    def run(self, *, limit: int = 500) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        for file_ref in self._iter_file_refs(limit=limit):
            validation = self.validate_file(file_ref["area"], file_ref["relative_path"])
            files.append(
                {
                    "area": file_ref["area"],
                    "relative_path": file_ref["relative_path"],
                    "name": file_ref.get("name"),
                    "metadata": validation.get("metadata", {}),
                    "issue_count": len(validation.get("issues", [])),
                    "ok": validation.get("ok", False),
                    "content_hash": validation.get("content_hash"),
                }
            )
            issues.extend(validation.get("issues", []))

        issues.extend(self._duplicate_issues(files))
        issues = self._sort_issues(issues)
        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        infos = [i for i in issues if i["severity"] == "info"]
        review = [i for i in issues if i["code"] in {"missing_last_reviewed", "stale_review", "invalid_last_reviewed"}]
        health_score = self._health_score(file_count=len(files), errors=len(errors), warnings=len(warnings), infos=len(infos))
        return {
            "kind": "knowledge_governance_report",
            "ok": not errors,
            "file_count": len(files),
            "issue_count": len(issues),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(infos),
            "review_recommended_count": len(review),
            "health_score": health_score,
            "grade": self._grade(health_score),
            "summary": self._summary(len(files), errors, warnings, infos),
            "files": files,
            "issues": issues,
        }

    def validate_metadata(self, metadata: dict[str, Any], *, area: str = "public", relative_path: str = "inline") -> dict[str, Any]:
        issues = self._issues_for_metadata(metadata, area=area, relative_path=relative_path)
        issues = self._sort_issues(issues)
        return {"ok": not any(i["severity"] == "error" for i in issues), "issues": issues, "metadata": metadata}

    def validate_file(self, area: str, relative_path: str) -> dict[str, Any]:
        payload = self.knowledge.show_file(area, relative_path, max_lines=10_000)
        metadata = payload.get("metadata", {})
        raw_text = ""
        body = payload.get("preview", "")
        file_path = self._file_path(area, relative_path)
        if file_path and file_path.exists() and file_path.is_file():
            raw_text = self._safe_read(file_path)
            body = strip_frontmatter(raw_text) if file_path.suffix.lower() == ".md" else raw_text
        issues = self._issues_for_metadata(metadata, area=area, relative_path=relative_path)
        issues.extend(self._issues_for_content(body, metadata=metadata, area=area, relative_path=relative_path, payload=payload))
        content_hash = hashlib.sha256(body.strip().encode("utf-8", errors="replace")).hexdigest() if body.strip() else ""
        issues = self._sort_issues(issues)
        return {
            "area": area,
            "relative_path": relative_path,
            "ok": not any(i["severity"] == "error" for i in issues),
            "issues": issues,
            "metadata": metadata,
            "content_hash": content_hash,
        }

    def validate_text(self, text: str, *, area: str = "public", relative_path: str = "inline.md") -> dict[str, Any]:
        metadata = normalize_metadata(area, relative_path, text)
        issues = self._issues_for_metadata(metadata, area=area, relative_path=relative_path)
        issues.extend(self._issues_for_content(strip_frontmatter(text), metadata=metadata, area=area, relative_path=relative_path, payload={}))
        return {"ok": not any(i["severity"] == "error" for i in issues), "issues": issues, "metadata": metadata}

    def _issues_for_metadata(self, metadata: dict[str, Any], *, area: str, relative_path: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        def add(severity: str, code: str, message: str) -> None:
            issues.append({"severity": severity, "code": code, "area": area, "relative_path": relative_path, "message": message})

        has_frontmatter = bool(metadata.get("has_frontmatter"))
        if not has_frontmatter:
            add("warning", "missing_frontmatter", "Markdown-Datei hat keinen YAML-Metadaten-Header.")
        for field_name in REQUIRED_FRONTMATTER_FIELDS:
            value = metadata.get(field_name)
            if field_name == "tags":
                if not value:
                    add("warning", "missing_tags", "tags fehlen; mindestens ein fachlicher Tag wird empfohlen.")
                continue
            if field_name == "cloud_allowed":
                if not has_frontmatter:
                    add("warning", "implicit_cloud_policy", "cloud_allowed fehlt im YAML-Header; Pandora nutzt nur den Ordner-Default.")
                continue
            if value in {None, ""}:
                add("warning", f"missing_{field_name}", f"{field_name} fehlt.")

        visibility = str(metadata.get("visibility") or "").strip()
        if visibility not in VALID_VISIBILITIES:
            add("error", "invalid_visibility", f"visibility muss eine der erlaubten Klassen sein: {sorted(VALID_VISIBILITIES)}")
        elif visibility != area:
            add("error", "visibility_area_mismatch", "visibility passt nicht zum Knowledge-Ordner.")

        cloud_allowed = bool(metadata.get("cloud_allowed"))
        if area == "private_local_only" and cloud_allowed:
            add("error", "private_cloud_allowed", "private_local_only darf niemals cloud_allowed=true setzen.")
        if area in {"public", "restricted_cloud_allowed"} and not cloud_allowed:
            add("warning", "cloud_blocked_in_cloud_area", "Datei liegt in Cloud-fähigem Bereich, setzt aber cloud_allowed=false.")
        if area == "restricted_cloud_allowed" and cloud_allowed and not metadata.get("summary"):
            add("warning", "restricted_missing_summary", "restricted_cloud_allowed sollte eine summary enthalten, damit Cloud-Kontext besser prüfbar ist.")

        priority = str(metadata.get("priority") or "normal").strip().lower()
        if priority not in VALID_PRIORITIES:
            add("warning", "invalid_priority", f"priority sollte eine der Klassen sein: {sorted(VALID_PRIORITIES)}")

        tags = metadata.get("tags") or []
        if not isinstance(tags, list):
            add("warning", "invalid_tags", "tags sollte eine Liste sein.")
            tags = []
        clean_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        if len(clean_tags) < MIN_TAGS:
            add("warning", "too_few_tags", "Mindestens ein aussagekräftiger Tag wird empfohlen.")
        for tag in clean_tags:
            if len(tag) > 40 or " " in tag:
                add("info", "tag_style", f"Tag '{tag}' ist lang oder enthält Leerzeichen; kurze Tags sind besser für Suche und Graph.")

        last_reviewed = str(metadata.get("last_reviewed") or "").strip()
        if not last_reviewed:
            add("warning", "missing_last_reviewed", "last_reviewed fehlt; Review empfohlen.")
        else:
            reviewed = self._parse_review_date(last_reviewed)
            if reviewed is None:
                add("warning", "invalid_last_reviewed", "last_reviewed sollte ISO-Format haben, z.B. 2026-06-09.")
            elif reviewed < datetime.now(UTC) - timedelta(days=REVIEW_MAX_AGE_DAYS):
                add("warning", "stale_review", f"last_reviewed ist älter als {REVIEW_MAX_AGE_DAYS} Tage.")
        return issues

    def _issues_for_content(self, text: str, *, metadata: dict[str, Any], area: str, relative_path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        def add(severity: str, code: str, message: str) -> None:
            issues.append({"severity": severity, "code": code, "area": area, "relative_path": relative_path, "message": message})

        stripped = (text or "").strip()
        words = [w for w in stripped.replace("#", " ").split() if w.strip()]
        if not stripped:
            add("warning", "empty_content", "Datei enthält keinen nutzbaren Inhalt.")
        elif len(words) < MIN_MEANINGFUL_WORDS:
            add("info", "very_short_content", f"Datei ist sehr kurz ({len(words)} Wörter); als Kontext eventuell wenig nützlich.")
        if len(stripped) > MAX_CONTEXT_CHARS_RECOMMENDED:
            add("warning", "large_context_file", "Datei ist sehr groß; später Chunking/Splitting einplanen.")
        if relative_path.lower().endswith(".txt"):
            add("info", "txt_without_metadata", "TXT-Dateien haben keine YAML-Metadaten; Markdown ist für Knowledge-Governance besser.")
        if area == "public" and any(term in stripped.lower() for term in ("passwort", "password", "api key", "token", "secret")):
            add("error", "possible_secret_in_public", "Möglicher Secret-/Passwort-Hinweis in public erkannt.")
        if bool(metadata.get("cloud_allowed")) and any(term in stripped.lower() for term in ("vertraulich", "confidential", "nur intern", "internal only")):
            add("warning", "cloud_allowed_confidential_terms", "Cloud-freigegebene Datei enthält vertrauliche Schlüsselwörter; bitte prüfen.")
        return issues


    def _sort_issues(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        order = {"error": 0, "warning": 1, "info": 2}
        return sorted(issues, key=lambda item: (order.get(item.get("severity", "info"), 9), item.get("area", ""), item.get("relative_path", ""), item.get("code", "")))

    def _duplicate_issues(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        by_title: dict[str, list[dict[str, Any]]] = {}
        by_hash: dict[str, list[dict[str, Any]]] = {}
        for file in files:
            title = str((file.get("metadata") or {}).get("title") or "").strip().lower()
            if title:
                by_title.setdefault(title, []).append(file)
            content_hash = file.get("content_hash")
            if content_hash:
                by_hash.setdefault(str(content_hash), []).append(file)
        for title, grouped in by_title.items():
            if len(grouped) > 1:
                for file in grouped:
                    issues.append({
                        "severity": "info",
                        "code": "duplicate_title",
                        "area": file["area"],
                        "relative_path": file["relative_path"],
                        "message": f"Titel kommt mehrfach vor ({len(grouped)} Dateien): {title}",
                    })
        for _content_hash, grouped in by_hash.items():
            if len(grouped) > 1:
                for file in grouped:
                    issues.append({
                        "severity": "warning",
                        "code": "duplicate_content",
                        "area": file["area"],
                        "relative_path": file["relative_path"],
                        "message": f"Inhalt scheint doppelt vorhanden zu sein ({len(grouped)} Dateien).",
                    })
        return issues

    def _iter_file_refs(self, *, limit: int) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for area in AREAS:
            area_payload = self.knowledge.list_area(area.name, limit=limit)
            for item in area_payload.get("files", []):
                refs.append({"area": area.name, "relative_path": item["relative_path"], "name": item.get("name"), "policy": area.policy})
                if len(refs) >= limit:
                    return refs
        return refs

    def _file_path(self, area: str, relative_path: str) -> Path | None:
        try:
            area_def = next(a for a in AREAS if a.name == area)
        except StopIteration:
            return None
        root = self.knowledge.root_dir / area_def.directory
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate

    def _safe_read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _parse_review_date(self, value: str) -> datetime | None:
        try:
            normalized = value.strip()
            if len(normalized) == 10:
                return datetime.fromisoformat(normalized).replace(tzinfo=UTC)
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None

    def _health_score(self, *, file_count: int, errors: int, warnings: int, infos: int) -> int:
        if file_count == 0:
            return 100
        score = 100 - errors * 18 - warnings * 6 - infos * 1
        return max(0, min(100, score))

    def _grade(self, score: int) -> str:
        if score >= 90:
            return "A"
        if score >= 75:
            return "B"
        if score >= 60:
            return "C"
        if score >= 40:
            return "D"
        return "E"

    def _summary(self, file_count: int, errors: list[dict[str, Any]], warnings: list[dict[str, Any]], infos: list[dict[str, Any]]) -> str:
        if file_count == 0:
            return "Keine Knowledge-Dateien gefunden."
        if errors:
            return f"{len(errors)} kritische Governance-Probleme gefunden. Bitte vor Cloud-Nutzung beheben."
        if warnings:
            return f"Keine kritischen Fehler, aber {len(warnings)} Warnungen gefunden. Review empfohlen."
        return "Knowledge Governance OK."
