from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR, USER_KNOWLEDGE_DIR
from .knowledge_editor import KnowledgeEditorService
from .knowledge_metadata import strip_frontmatter
from .obsidian_import_candidates import ObsidianImportCandidateService
from .obsidian_vault import ObsidianSafetyError, ObsidianVaultService

OBSIDIAN_IMPORT_EXECUTIONS_DIR = PROPOSALS_DIR / "obsidian_import_executions"
IMPORT_READY_STATUS = "accepted_for_next_step"


@dataclass
class ObsidianImportExecutionService:
    """Plan and perform controlled Obsidian -> Pandora Knowledge imports.

    This service is intentionally strict:
    - candidates must be accepted for the next step before execution
    - execution requires explicit confirm=True
    - writes are limited to user_knowledge/ through KnowledgeEditorService
    - existing files are not overwritten unless explicitly requested
    - Obsidian files are never modified, moved or deleted
    """

    candidates: ObsidianImportCandidateService = field(default_factory=ObsidianImportCandidateService)
    editor: KnowledgeEditorService = field(default_factory=KnowledgeEditorService)
    vault: ObsidianVaultService = field(default_factory=ObsidianVaultService)
    executions_dir: Path = OBSIDIAN_IMPORT_EXECUTIONS_DIR

    def status(self) -> dict[str, Any]:
        executions = self.list_executions(limit=10000)["executions"]
        return {
            "kind": "obsidian_import_execution_status",
            "version": "mvp-23.5.6-obsidian-import-execution-plan",
            "generated_at": datetime.now(UTC).isoformat(),
            "execution_count": len(executions),
            "executions_dir": str(self.executions_dir),
            "safety": self._safety(),
        }

    def build_plan(self, candidate_id: str, *, overwrite: bool = False) -> dict[str, Any]:
        detail = self.candidates.show(candidate_id)
        if not detail.get("found"):
            return {"kind": "obsidian_import_execution_plan", "ok": False, "candidate_id": candidate_id, "reason": "candidate not found"}
        candidate = detail["candidate"]
        source = self._read_source(candidate.get("source_relative_path", ""))
        target_area = str(candidate.get("target_area") or "private_local_only")
        target_rel = self._target_relative_path(candidate)
        target_abs = self._target_abs_path(target_area, target_rel)
        review_status = str(candidate.get("status") or "pending_review")
        target_exists = target_abs.exists()
        errors: list[str] = []
        warnings: list[str] = []
        if not source.get("ok"):
            errors.append(source.get("error") or "source cannot be read")
        if review_status != IMPORT_READY_STATUS:
            warnings.append(f"candidate status is {review_status!r}; execution requires {IMPORT_READY_STATUS!r}")
        if target_exists and not overwrite:
            errors.append("target file already exists; use overwrite=true only after manual review")
        if target_area == "private_local_only" and candidate.get("proposed_metadata", {}).get("cloud_allowed") is True:
            warnings.append("private_local_only target forces cloud_allowed=false during import")
        allowed = not errors and review_status == IMPORT_READY_STATUS
        return {
            "kind": "obsidian_import_execution_plan",
            "ok": True,
            "candidate_id": candidate_id,
            "allowed_to_execute": allowed,
            "requires_confirm": True,
            "candidate_status": review_status,
            "source": {
                "type": "obsidian",
                "relative_path": candidate.get("source_relative_path"),
                "readable": source.get("ok", False),
                "content_chars": source.get("content_chars", 0),
            },
            "target": {
                "area": target_area,
                "relative_path": target_rel,
                "absolute_path": str(target_abs),
                "exists": target_exists,
                "overwrite_requested": overwrite,
            },
            "proposed_metadata": self._metadata_for_import(candidate),
            "steps": [
                "read Obsidian source note",
                "strip existing Obsidian frontmatter from imported body",
                "compose Pandora Knowledge metadata",
                "write Markdown via KnowledgeEditorService",
                "run Knowledge Governance validation",
                "write import execution audit record",
            ],
            "errors": errors,
            "warnings": warnings,
            "safety": self._safety(),
        }

    def execute(self, candidate_id: str, *, confirm: bool = False, overwrite: bool = False, executed_by: str = "user") -> dict[str, Any]:
        plan = self.build_plan(candidate_id, overwrite=overwrite)
        if not confirm:
            return {"kind": "obsidian_import_execution", "ok": False, "candidate_id": candidate_id, "reason": "confirm=true is required", "plan": plan}
        if not plan.get("allowed_to_execute"):
            return {"kind": "obsidian_import_execution", "ok": False, "candidate_id": candidate_id, "reason": "plan is not executable", "plan": plan}
        detail = self.candidates.show(candidate_id)
        candidate = detail["candidate"]
        source = self._read_source(str(candidate.get("source_relative_path") or ""))
        if not source.get("ok"):
            return {"kind": "obsidian_import_execution", "ok": False, "candidate_id": candidate_id, "reason": source.get("error"), "plan": plan}
        area = plan["target"]["area"]
        relative_path = plan["target"]["relative_path"]
        metadata = self._metadata_for_import(candidate)
        body = self._body_for_import(source.get("content", ""), candidate)
        saved = self.editor.save_file(area=area, relative_path=relative_path, metadata=metadata, body=body, overwrite=overwrite)
        audit = {
            "kind": "obsidian_import_execution_audit",
            "candidate_id": candidate_id,
            "executed_at": datetime.now(UTC).isoformat(),
            "executed_by": executed_by,
            "source_relative_path": candidate.get("source_relative_path"),
            "target_area": area,
            "target_relative_path": relative_path,
            "overwrite": overwrite,
            "saved": saved,
            "obsidian_modified": False,
            "obsidian_deleted": False,
            "user_knowledge_written": True,
        }
        audit_path = self._write_audit(candidate_id, audit)
        return {
            "kind": "obsidian_import_execution",
            "ok": True,
            "candidate_id": candidate_id,
            "target": plan["target"],
            "saved": saved,
            "audit_path": str(audit_path),
            "safety": self._safety(),
        }

    def list_executions(self, *, limit: int = 200) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if self.executions_dir.exists():
            for path in sorted(self.executions_dir.rglob("execution.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        data["source_file"] = str(path)
                        items.append(data)
                except (OSError, json.JSONDecodeError):
                    continue
                if len(items) >= limit:
                    break
        return {"kind": "obsidian_import_execution_list", "count": len(items), "executions": items}

    def _read_source(self, relative_path: str) -> dict[str, Any]:
        try:
            path = self.vault._safe_vault_file(relative_path)  # guarded by ObsidianVaultService
            text = path.read_text(encoding="utf-8", errors="replace")
            return {"ok": True, "relative_path": relative_path, "content": text, "content_chars": len(text)}
        except (OSError, ObsidianSafetyError) as exc:
            return {"ok": False, "relative_path": relative_path, "error": str(exc), "content_chars": 0}

    def _target_relative_path(self, candidate: dict[str, Any]) -> str:
        folder = str(candidate.get("suggested_folder") or "obsidian").strip().strip("/\\") or "obsidian"
        file_name = str(candidate.get("suggested_file_name") or "obsidian_note.md").strip().strip("/\\") or "obsidian_note.md"
        if not file_name.lower().endswith(".md"):
            file_name += ".md"
        return f"{folder}/{file_name}".replace("\\", "/")

    def _target_abs_path(self, area: str, relative_path: str) -> Path:
        root = USER_KNOWLEDGE_DIR / area
        return (root / relative_path).resolve()

    def _metadata_for_import(self, candidate: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(candidate.get("proposed_metadata") or {})
        target_area = str(candidate.get("target_area") or metadata.get("visibility") or "private_local_only")
        metadata["visibility"] = target_area
        if target_area == "private_local_only":
            metadata["cloud_allowed"] = False
        else:
            metadata.setdefault("cloud_allowed", True)
        metadata.setdefault("title", candidate.get("title") or "Obsidian Import")
        metadata.setdefault("tags", candidate.get("tags") or ["obsidian"])
        metadata.setdefault("priority", candidate.get("priority") or "medium")
        metadata.setdefault("owner", "user")
        metadata["last_reviewed"] = datetime.now(UTC).date().isoformat()
        metadata["source"] = "obsidian"
        metadata["source_path"] = candidate.get("source_relative_path")
        metadata["imported_by"] = "pandora"
        return metadata

    def _body_for_import(self, source_text: str, candidate: dict[str, Any]) -> str:
        body = strip_frontmatter(source_text or "").strip()
        source_path = candidate.get("source_relative_path") or "unknown"
        note = f"\n\n---\n\n_Importiert aus Obsidian: `{source_path}`._\n"
        return body + note

    def _write_audit(self, candidate_id: str, audit: dict[str, Any]) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in candidate_id)[:120]
        target_dir = self.executions_dir / safe
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "execution.json"
        target.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return target

    def _safety(self) -> dict[str, Any]:
        return {
            "requires_accepted_candidate": True,
            "requires_confirm": True,
            "writes_user_knowledge_only": True,
            "obsidian_read_only": True,
            "obsidian_delete_allowed": False,
            "obsidian_move_allowed": False,
            "overwrite_default": False,
        }
