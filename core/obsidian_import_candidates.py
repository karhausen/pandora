from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, USER_KNOWLEDGE_DIR
from .obsidian_vault import ObsidianSafetyError, ObsidianVaultService

OBSIDIAN_IMPORT_CANDIDATES_DIR = PROPOSALS_DIR / "obsidian_import_candidates"
VALID_DECISIONS = {"reviewed", "accepted_for_next_step", "rejected", "needs_work", "deferred"}


@dataclass(frozen=True)
class ObsidianImportCandidate:
    """Reviewable proposal to import an Obsidian note into Pandora Knowledge.

    The candidate is deliberately non-executing. It contains enough metadata for
    a user to decide whether a Vault note should later be copied into
    `user_knowledge/`, but it never performs the import automatically.
    """

    id: str
    title: str
    source_relative_path: str
    source_type: str = "obsidian"
    target_area: str = "private_local_only"
    suggested_folder: str = "obsidian"
    suggested_file_name: str = "note.md"
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    priority: str = "medium"
    reason: str = "Obsidian note may be useful for Pandora Knowledge Base."
    summary: str = ""
    status: str = "pending_review"
    created_at: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "obsidian_import_candidate",
            "title": self.title,
            "source_type": self.source_type,
            "source_relative_path": self.source_relative_path,
            "target_area": self.target_area,
            "suggested_folder": self.suggested_folder,
            "suggested_file_name": self.suggested_file_name,
            "proposed_target_path": f"user_knowledge/{self.target_area}/{self.suggested_folder}/{self.suggested_file_name}",
            "proposed_metadata": {
                "title": self.title,
                "tags": self.tags or ["obsidian"],
                "visibility": self.target_area,
                "cloud_allowed": self.target_area != "private_local_only",
                "priority": self.priority,
                "owner": "user",
                "last_reviewed": datetime.now(UTC).date().isoformat(),
                "source": "obsidian",
                "source_path": self.source_relative_path,
            },
            "tags": self.tags,
            "wikilinks": self.wikilinks,
            "priority": self.priority,
            "reason": self.reason,
            "summary": self.summary,
            "status": self.status,
            "created_at": self.created_at,
            "requires_user_review": True,
            "auto_import": False,
            "auto_write_knowledge": False,
            "risk": "low",
            "evidence": self.evidence,
        }


class ObsidianImportCandidateService:
    """Create review-only import candidates from an Obsidian vault.

    The service reads the vault, proposes target locations and metadata for
    Pandora's internal Knowledge Base and writes proposal JSON files. It does
    not copy, delete, move or modify Obsidian files and does not write into
    `user_knowledge/`.
    """

    def __init__(
        self,
        *,
        candidates_dir: Path = OBSIDIAN_IMPORT_CANDIDATES_DIR,
        vault: ObsidianVaultService | None = None,
    ) -> None:
        self.candidates_dir = candidates_dir
        self.vault = vault or ObsidianVaultService()

    def status(self) -> dict[str, Any]:
        listing = self.list_candidates(include_reviewed=True, limit=10000)
        candidates = listing["candidates"]
        return {
            "kind": "obsidian_import_candidates_status",
            "version": "mvp-23.5.5-obsidian-knowledge-import-candidates",
            "generated_at": datetime.now(UTC).isoformat(),
            "candidates_dir": str(self.candidates_dir),
            "exists": self.candidates_dir.exists(),
            "candidate_count": len(candidates),
            "open_count": sum(1 for item in candidates if self._is_open_status(item.get("status"))),
            "counts_by_target_area": self._count_by(candidates, "target_area"),
            "counts_by_status": self._count_by(candidates, "status"),
            "safety": self._safety(),
        }

    def build(self, *, query: str | None = None, limit: int = 50, write: bool = True) -> dict[str, Any]:
        source = self._source_records(query=query, limit=limit)
        created_at = datetime.now(UTC).isoformat()
        candidates = [self._candidate_from_record(record, created_at=created_at).as_dict() for record in source]
        if write:
            self.candidates_dir.mkdir(parents=True, exist_ok=True)
            for candidate in candidates:
                self._write_candidate(candidate)
        return {
            "kind": "obsidian_import_candidates_build_report",
            "version": "mvp-23.5.5-obsidian-knowledge-import-candidates",
            "created_at": created_at,
            "query": query,
            "write": write,
            "source_count": len(source),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "safety": self._safety(),
        }

    def list_candidates(
        self,
        *,
        include_reviewed: bool = False,
        limit: int = 200,
        target_area: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        if self.candidates_dir.exists():
            for path in sorted(self.candidates_dir.rglob("proposal.json")):
                data = self._read_json(path)
                if not data:
                    continue
                data = self._with_review_state(data, path)
                if not include_reviewed and not self._is_open_status(data.get("status")):
                    continue
                if target_area and data.get("target_area") != target_area:
                    continue
                if status and data.get("status") != status:
                    continue
                if query:
                    haystack = " ".join(str(data.get(k, "")) for k in ["id", "title", "source_relative_path", "suggested_folder", "summary", "reason"]).lower()
                    haystack += " " + " ".join(data.get("tags") or []).lower()
                    if query.lower() not in haystack:
                        continue
                data["source_file"] = str(path)
                candidates.append(data)
        candidates.sort(key=lambda item: (self._priority_rank(item.get("priority")), item.get("created_at") or ""), reverse=True)
        return {
            "kind": "obsidian_import_candidates_list",
            "include_reviewed": include_reviewed,
            "filters": {"target_area": target_area, "status": status, "query": query},
            "total_count": len(candidates),
            "count": min(len(candidates), limit),
            "candidates": candidates[:limit],
            "summary": {
                "by_target_area": self._count_by(candidates, "target_area"),
                "by_status": self._count_by(candidates, "status"),
                "by_priority": self._count_by(candidates, "priority"),
            },
            "safety": self._safety(),
        }

    def show(self, candidate_id: str) -> dict[str, Any]:
        for candidate in self.list_candidates(include_reviewed=True, limit=10000)["candidates"]:
            if candidate.get("id") == candidate_id:
                preview = self._source_preview(candidate.get("source_relative_path", ""))
                return {
                    "kind": "obsidian_import_candidate_detail",
                    "found": True,
                    "candidate": candidate,
                    "source_preview": preview,
                    "safety": self._safety(),
                }
        return {"kind": "obsidian_import_candidate_detail", "found": False, "id": candidate_id}

    def decide(self, candidate_id: str, *, decision: str, note: str | None = None, decided_by: str = "user") -> dict[str, Any]:
        if decision not in VALID_DECISIONS:
            return {"kind": "obsidian_import_candidate_decision", "ok": False, "reason": f"decision must be one of {sorted(VALID_DECISIONS)}", "candidate_id": candidate_id}
        detail = self.show(candidate_id)
        if not detail.get("found"):
            return {"kind": "obsidian_import_candidate_decision", "ok": False, "reason": "candidate not found", "candidate_id": candidate_id}
        candidate = detail["candidate"]
        source_file = Path(candidate["source_file"])
        payload = {
            "kind": "review_state",
            "item_id": candidate_id,
            "decision": decision,
            "note": note,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "reviewed_by": decided_by,
            "auto_changes_made": False,
            "import_performed": False,
            "requires_separate_import_step": decision == "accepted_for_next_step",
        }
        state_path = source_file.parent / "review_state.json"
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {
            "kind": "obsidian_import_candidate_decision",
            "ok": True,
            "candidate_id": candidate_id,
            "decision": decision,
            "written_to": str(state_path),
            "state": payload,
            "safety": self._safety(),
        }

    def _source_records(self, *, query: str | None, limit: int) -> list[dict[str, Any]]:
        if query and query.strip():
            records = self.vault.search(query, limit=limit, include_content=False).get("results", [])
        else:
            records = self.vault.index(limit=limit, write=False).get("files", [])
        filtered: list[dict[str, Any]] = []
        inbox_prefix = self.vault.config.inbox_dir.strip("/\\") + "/"
        for record in records:
            rel = str(record.get("relative_path") or "")
            if not rel or rel.startswith(inbox_prefix) or rel.startswith(".obsidian/"):
                continue
            if int(record.get("word_count") or 0) < 5:
                continue
            filtered.append(record)
        return filtered[:limit]

    def _candidate_from_record(self, record: dict[str, Any], *, created_at: str) -> ObsidianImportCandidate:
        rel = str(record.get("relative_path") or "unknown.md")
        title = str(record.get("title") or Path(rel).stem)
        tags = [str(tag) for tag in (record.get("tags") or [])][:12]
        wikilinks = [str(link) for link in (record.get("wikilinks") or [])][:12]
        target_area = self._suggest_target_area(tags, rel)
        folder = self._suggest_folder(tags, rel, wikilinks)
        file_name = self._safe_filename(title) + ".md"
        candidate_id = "obsidian_import:" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]
        reason = self._reason(tags, rel, target_area)
        priority = "high" if tags or wikilinks else "medium"
        return ObsidianImportCandidate(
            id=candidate_id,
            title=title,
            source_relative_path=rel,
            target_area=target_area,
            suggested_folder=folder,
            suggested_file_name=file_name,
            tags=tags,
            wikilinks=wikilinks,
            priority=priority,
            reason=reason,
            summary=str(record.get("excerpt") or "")[:500],
            created_at=created_at,
            evidence={
                "word_count": record.get("word_count"),
                "modified_at": record.get("modified_at"),
                "sha256": record.get("sha256"),
                "obsidian_cloud_allowed": self.vault.config.cloud_allowed,
            },
        )

    def _suggest_target_area(self, tags: list[str], rel: str) -> str:
        text = (" ".join(tags) + " " + rel).lower()
        private_markers = {"private", "privat", "personal", "persoenlich", "firma", "intern", "secret", "confidential", "vertraulich"}
        public_markers = {"public", "öffentlich", "open", "docs", "reference"}
        if any(marker in text for marker in private_markers):
            return "private_local_only"
        if self.vault.config.cloud_allowed and any(marker in text for marker in public_markers):
            return "public"
        if self.vault.config.cloud_allowed:
            return "restricted_cloud_allowed"
        return "private_local_only"

    def _suggest_folder(self, tags: list[str], rel: str, wikilinks: list[str]) -> str:
        candidates = [tag for tag in tags if tag.lower() not in {"public", "private", "privat", "obsidian", "pandora"}]
        if candidates:
            return self._safe_segment(candidates[0])
        if wikilinks:
            return self._safe_segment(wikilinks[0].split("|", 1)[0].split("/", 1)[0])
        parent = Path(rel).parent.as_posix()
        if parent and parent != ".":
            return self._safe_segment(parent.split("/", 1)[0])
        return "obsidian"

    def _reason(self, tags: list[str], rel: str, target_area: str) -> str:
        if tags:
            return f"Vault note has tags ({', '.join(tags[:5])}) and may enrich Pandora Knowledge. Suggested target: {target_area}."
        return f"Vault note '{rel}' may enrich Pandora Knowledge. Suggested target: {target_area}."

    def _source_preview(self, relative_path: str) -> dict[str, Any]:
        try:
            result = self.vault.search(relative_path, limit=1, include_content=True)
            for item in result.get("results", []):
                if item.get("relative_path") == relative_path:
                    content = item.get("content", "")
                    return {"ok": True, "relative_path": relative_path, "content_preview": content[:4000], "content_chars": len(content)}
        except ObsidianSafetyError as exc:
            return {"ok": False, "relative_path": relative_path, "error": str(exc)}
        return {"ok": False, "relative_path": relative_path, "error": "source preview not found"}

    def _write_candidate(self, candidate: dict[str, Any]) -> None:
        directory = self.candidates_dir / self._safe_id(str(candidate["id"]))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "proposal.json"
        existing = self._read_json(path) or {}
        if existing.get("status") in {"reviewed", "accepted_for_next_step", "rejected", "needs_work"}:
            candidate["status"] = existing.get("status")
            candidate["review_locked"] = True
        path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _with_review_state(self, data: dict[str, Any], proposal_path: Path) -> dict[str, Any]:
        state = self._read_json(proposal_path.parent / "review_state.json") or {}
        result = dict(data)
        if state.get("decision"):
            result["status"] = state["decision"]
            result["review_state"] = state
            result["reviewed_at"] = state.get("reviewed_at")
            result["reviewed_by"] = state.get("reviewed_by")
        else:
            result["status"] = result.get("status") or "pending_review"
        return result

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _is_open_status(self, status: Any) -> bool:
        return str(status or "pending_review") in {"pending", "pending_review", "needs_work", "deferred"}

    def _priority_rank(self, priority: Any) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(str(priority), 0)

    def _count_by(self, items: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            value = str(item.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _safe_id(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value).lower()[:120]

    def _safe_filename(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß _.-]", "_", value).strip().strip(".")
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned[:90] or "obsidian_note"

    def _safe_segment(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9ÄÖÜäöüß _./-]", "_", value).strip().strip("./\\")
        cleaned = cleaned.replace("..", "_")
        return cleaned[:120] or "obsidian"

    def _safety(self) -> dict[str, Any]:
        return {
            "observe_only": True,
            "requires_user_review": True,
            "auto_import": False,
            "auto_write_knowledge": False,
            "reads_obsidian": True,
            "writes_reviewable_json_only": True,
            "candidate_output_dir": str(self.candidates_dir),
            "knowledge_base_dir": str(USER_KNOWLEDGE_DIR),
        }
