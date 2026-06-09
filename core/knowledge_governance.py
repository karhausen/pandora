from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from typing import Any

from .knowledge_metadata import normalize_metadata
from .user_knowledge_base import AREAS, UserKnowledgeBaseService

VALID_VISIBILITIES = {"public", "restricted_cloud_allowed", "private_local_only"}
VALID_PRIORITIES = {"low", "normal", "high", "critical"}
REVIEW_MAX_AGE_DAYS = 180


@dataclass
class KnowledgeGovernanceService:
    """Governance/audit service for user-managed knowledge files.

    It checks frontmatter metadata, cloud-policy consistency, stale reviews and
    obvious policy mismatches. It is read-only and never moves or edits user files.
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
            "review_recommended_count": report["review_recommended_count"],
        }

    def metadata_index(self, *, limit: int = 500) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for area in AREAS:
            area_payload = self.knowledge.list_area(area.name, limit=limit)
            for item in area_payload.get("files", []):
                shown = self.knowledge.show_file(area.name, item["relative_path"], max_lines=500)
                files.append({
                    "area": area.name,
                    "relative_path": item["relative_path"],
                    "name": item.get("name"),
                    "policy": area.policy,
                    "metadata": shown.get("metadata", {}),
                    "governance": self.validate_file(area.name, item["relative_path"]),
                })
                if len(files) >= limit:
                    return {"kind": "knowledge_metadata_index", "count": len(files), "files": files, "truncated": True}
        return {"kind": "knowledge_metadata_index", "count": len(files), "files": files, "truncated": False}

    def run(self, *, limit: int = 500) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        file_count = 0
        for area in AREAS:
            area_payload = self.knowledge.list_area(area.name, limit=limit)
            for item in area_payload.get("files", []):
                file_count += 1
                validation = self.validate_file(area.name, item["relative_path"])
                issues.extend(validation.get("issues", []))
                if file_count >= limit:
                    break
        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        review = [i for i in issues if i["code"] in {"missing_last_reviewed", "stale_review"}]
        return {
            "kind": "knowledge_governance_report",
            "ok": not errors,
            "file_count": file_count,
            "issue_count": len(issues),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "review_recommended_count": len(review),
            "issues": issues,
        }

    def validate_metadata(self, metadata: dict[str, Any], *, area: str = "public", relative_path: str = "inline") -> dict[str, Any]:
        issues = self._issues_for_metadata(metadata, area=area, relative_path=relative_path)
        return {"ok": not any(i["severity"] == "error" for i in issues), "issues": issues, "metadata": metadata}

    def validate_file(self, area: str, relative_path: str) -> dict[str, Any]:
        payload = self.knowledge.show_file(area, relative_path, max_lines=1000)
        metadata = payload.get("metadata", {})
        issues = self._issues_for_metadata(metadata, area=area, relative_path=relative_path)
        return {"area": area, "relative_path": relative_path, "ok": not any(i["severity"] == "error" for i in issues), "issues": issues}

    def validate_text(self, text: str, *, area: str = "public", relative_path: str = "inline.md") -> dict[str, Any]:
        metadata = normalize_metadata(area, relative_path, text)
        return self.validate_metadata(metadata, area=area, relative_path=relative_path)

    def _issues_for_metadata(self, metadata: dict[str, Any], *, area: str, relative_path: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        def add(severity: str, code: str, message: str) -> None:
            issues.append({"severity": severity, "code": code, "area": area, "relative_path": relative_path, "message": message})

        if not metadata.get("has_frontmatter"):
            add("warning", "missing_frontmatter", "Markdown-Datei hat keinen YAML-Metadaten-Header.")
        if not metadata.get("title"):
            add("warning", "missing_title", "title fehlt.")
        visibility = metadata.get("visibility")
        if visibility not in VALID_VISIBILITIES:
            add("error", "invalid_visibility", f"visibility muss eine der erlaubten Klassen sein: {sorted(VALID_VISIBILITIES)}")
        elif visibility != area:
            add("error", "visibility_area_mismatch", "visibility passt nicht zum Knowledge-Ordner.")
        cloud_allowed = bool(metadata.get("cloud_allowed"))
        if area == "private_local_only" and cloud_allowed:
            add("error", "private_cloud_allowed", "private_local_only darf niemals cloud_allowed=true setzen.")
        if area in {"public", "restricted_cloud_allowed"} and not cloud_allowed:
            add("warning", "cloud_blocked_in_cloud_area", "Datei liegt in Cloud-fähigem Bereich, setzt aber cloud_allowed=false.")
        priority = metadata.get("priority") or "normal"
        if priority not in VALID_PRIORITIES:
            add("warning", "invalid_priority", f"priority sollte eine der Klassen sein: {sorted(VALID_PRIORITIES)}")
        last_reviewed = metadata.get("last_reviewed") or ""
        if not last_reviewed:
            add("warning", "missing_last_reviewed", "last_reviewed fehlt; Review empfohlen.")
        else:
            try:
                reviewed = datetime.fromisoformat(str(last_reviewed)).replace(tzinfo=UTC)
                if reviewed < datetime.now(UTC) - timedelta(days=REVIEW_MAX_AGE_DAYS):
                    add("warning", "stale_review", f"last_reviewed ist älter als {REVIEW_MAX_AGE_DAYS} Tage.")
            except ValueError:
                add("warning", "invalid_last_reviewed", "last_reviewed sollte ISO-Format haben, z.B. 2026-06-09.")
        return issues
