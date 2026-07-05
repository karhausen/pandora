from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import ast
import hashlib
import json

from core.proposal_queue import UnifiedProposalQueueManager
from core.genome import EvolutionService

ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = ROOT / "memory" / "core_evolution"
HISTORY_STORE = STORE_DIR / "history.json"

CORE_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "data", "memory", "tool_proposals", "logs", "dist", "build", ".git", ".venv", "venv"}
RISKY_IMPORTS = {"subprocess", "os", "shutil", "socket", "requests", "urllib", "httpx", "sqlite3"}
SAFETY_KEYWORDS = {"activate", "rollback", "delete", "remove", "write", "exec", "eval", "subprocess", "shell"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(*parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class CoreEvolutionManager:
    """Controlled Core Evolution for Pandora.

    This manager analyzes the Pandora core and generates review-only
    refactoring candidates. It never edits, replaces, activates or deletes core
    files. Any proposed work must pass Proposal Queue review and explicit user
    approval.
    """

    root: Path = field(default_factory=lambda: ROOT)
    history_store: Path = field(default_factory=lambda: HISTORY_STORE)
    VERSION = "29.5"

    def status(self) -> dict[str, Any]:
        health = self.health(limit=2000)
        proposals = self.proposals(limit=2000, enqueue=False)
        return {
            "kind": "core_evolution_status",
            "mvp": self.VERSION,
            "ok": True,
            "enabled": True,
            "mode": "analysis_and_proposal_only",
            "core_file_count": health.get("core_file_count", 0),
            "health_score": health.get("health_score", 100),
            "grade": health.get("grade", "A"),
            "risk_hotspot_count": len(health.get("risk_hotspots", [])),
            "refactoring_candidate_count": len(self.refactoring(limit=2000).get("candidates", [])),
            "proposal_candidate_count": proposals.get("count", 0),
            "policy": "Core Evolution analyzes and proposes. It never changes core files automatically.",
            "requires_user_approval": True,
            "activates_changes": False,
            "available_commands": ["status", "health", "analysis", "refactoring", "proposals", "enqueue", "history"],
        }

    def health(self, *, limit: int = 2000) -> dict[str, Any]:
        files = self._core_files(limit=limit)
        analyses = [self._analyze_file(path) for path in files]
        issue_count = sum(len(a["issues"]) for a in analyses)
        risk_hotspots = [a for a in analyses if a["risk_score"] >= 45 or a["issues"]]
        complexity_hotspots = [a for a in analyses if a["complexity_score"] >= 70]
        score = max(0, min(100, 100 - min(45, issue_count * 2) - min(25, len(complexity_hotspots))))
        return {
            "kind": "core_evolution_health",
            "mvp": self.VERSION,
            "ok": True,
            "core_file_count": len(files),
            "health_score": score,
            "grade": self._grade(score),
            "issue_count": issue_count,
            "risk_hotspot_count": len(risk_hotspots),
            "complexity_hotspot_count": len(complexity_hotspots),
            "risk_hotspots": sorted(risk_hotspots, key=lambda x: (-x["risk_score"], x["relative_path"]))[:50],
            "complexity_hotspots": sorted(complexity_hotspots, key=lambda x: (-x["complexity_score"], x["relative_path"]))[:50],
            "read_only": True,
            "activates_changes": False,
        }

    def analysis(self, *, limit: int = 2000, query: str | None = None) -> dict[str, Any]:
        files = self._core_files(limit=limit)
        analyses = [self._analyze_file(path) for path in files]
        if query:
            q = query.lower()
            analyses = [a for a in analyses if q in a["relative_path"].lower() or q in " ".join(a.get("issues", [])).lower()]
        return {
            "kind": "core_evolution_analysis",
            "mvp": self.VERSION,
            "ok": True,
            "count": len(analyses),
            "files": sorted(analyses, key=lambda x: (-x["risk_score"], -x["complexity_score"], x["relative_path"])),
            "read_only": True,
        }

    def refactoring(self, *, limit: int = 2000, min_severity: str = "warning") -> dict[str, Any]:
        severity_rank = {"info": 0, "warning": 1, "error": 2}
        min_rank = severity_rank.get(min_severity, 1)
        candidates: list[dict[str, Any]] = []
        for item in self.analysis(limit=limit).get("files", []):
            severity = self._severity(item)
            if severity_rank.get(severity, 0) < min_rank:
                continue
            if not item["issues"] and item["complexity_score"] < 70:
                continue
            candidates.append({
                "candidate_id": f"core_refactor_{_id(item['relative_path'], ','.join(item['issues']))}",
                "type": "core_refactoring_candidate",
                "proposal_type": "core",
                "title": f"Review core file {item['relative_path']}",
                "description": self._candidate_description(item),
                "severity": severity,
                "priority": self._priority(item),
                "confidence": 0.75 if item["issues"] else 0.55,
                "impact": "high" if severity == "error" else "medium",
                "risk": "high" if item["risk_score"] >= 70 else "medium",
                "source": "core_evolution",
                "evidence": item,
                "recommendation": self._recommendation(item),
                "requires_user_approval": True,
                "activates_changes": False,
            })
        return {
            "kind": "core_evolution_refactoring_candidates",
            "mvp": self.VERSION,
            "ok": True,
            "count": len(candidates),
            "candidates": sorted(candidates, key=lambda c: (-c["priority"], c["title"])),
            "policy": "Candidates are advisory only. No core file is modified.",
        }

    def proposals(self, *, limit: int = 2000, min_severity: str = "warning", enqueue: bool = False) -> dict[str, Any]:
        candidates = self.refactoring(limit=limit, min_severity=min_severity).get("candidates", [])
        proposals: list[dict[str, Any]] = []
        enqueued: list[dict[str, Any]] = []
        factory = EvolutionService()
        queue = UnifiedProposalQueueManager()
        for candidate in candidates:
            payload = {
                "type": "core",
                "title": candidate["title"],
                "description": candidate["description"],
                "source": "core_evolution",
                "priority": candidate["priority"],
                "confidence": candidate["confidence"],
                "impact": candidate["impact"],
                "risk": candidate["risk"],
                "payload": {"candidate": candidate, "policy": "review_only_no_activation"},
            }
            proposal = factory.factory_create(payload).get("proposal")
            proposals.append(proposal)
            if enqueue:
                enqueued.append(queue.enqueue(proposal))
        result = {
            "kind": "core_evolution_proposals",
            "mvp": self.VERSION,
            "ok": True,
            "count": len(proposals),
            "proposals": proposals,
            "enqueued": enqueued,
            "activates_changes": False,
            "requires_user_approval": True,
        }
        if enqueue:
            self._append_history({"action": "enqueue", "count": len(enqueued), "min_severity": min_severity})
        return result

    def enqueue(self, *, limit: int = 50, min_severity: str = "warning") -> dict[str, Any]:
        return self.proposals(limit=limit, min_severity=min_severity, enqueue=True)

    def history(self, *, limit: int = 50) -> dict[str, Any]:
        items = self._read_history()
        return {"kind": "core_evolution_history", "mvp": self.VERSION, "count": len(items[-limit:]), "history": list(reversed(items[-limit:]))}

    def _core_files(self, *, limit: int) -> list[Path]:
        roots = [self.root / "core", self.root / "main.py"]
        files: list[Path] = []
        for base in roots:
            if base.is_file():
                files.append(base)
            elif base.exists():
                for path in base.rglob("*.py"):
                    rel_parts = path.relative_to(self.root).parts
                    if any(part in CORE_EXCLUDE_DIRS for part in rel_parts):
                        continue
                    files.append(path)
        return sorted(files)[: max(0, int(limit))]

    def _analyze_file(self, path: Path) -> dict[str, Any]:
        rel = path.relative_to(self.root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        issues: list[str] = []
        imports: list[str] = []
        functions = classes = 0
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    functions += 1
                    if len(getattr(node, "body", [])) > 80:
                        issues.append(f"large_function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    classes += 1
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module.split(".")[0])
                elif isinstance(node, ast.Call):
                    name = self._call_name(node)
                    if name in {"eval", "exec"}:
                        issues.append(f"unsafe_call:{name}")
        except SyntaxError as exc:
            issues.append(f"syntax_error:{exc.lineno}")
        risky_imports = sorted(set(imports) & RISKY_IMPORTS)
        if risky_imports:
            issues.append("risky_imports:" + ",".join(risky_imports))
        lower = text.lower()
        keyword_hits = sorted(k for k in SAFETY_KEYWORDS if k in lower)
        complexity_score = min(100, len(lines) // 8 + functions * 2 + classes * 3)
        risk_score = min(100, len(risky_imports) * 12 + len(keyword_hits) * 4 + len(issues) * 8 + (20 if rel in {"main.py", "core/api.py"} else 0))
        return {
            "relative_path": rel,
            "lines": len(lines),
            "functions": functions,
            "classes": classes,
            "complexity_score": complexity_score,
            "risk_score": risk_score,
            "risky_imports": risky_imports,
            "safety_keyword_hits": keyword_hits[:20],
            "issues": sorted(set(issues)),
            "sha1": hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12],
        }

    def _call_name(self, node: ast.Call) -> str:
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return ""

    def _severity(self, item: dict[str, Any]) -> str:
        if item["risk_score"] >= 70 or any(str(i).startswith("syntax_error") or str(i).startswith("unsafe_call") for i in item["issues"]):
            return "error"
        if item["risk_score"] >= 40 or item["complexity_score"] >= 70 or item["issues"]:
            return "warning"
        return "info"

    def _priority(self, item: dict[str, Any]) -> int:
        return max(20, min(95, 35 + item["risk_score"] // 2 + item["complexity_score"] // 3 + len(item["issues"]) * 3))

    def _recommendation(self, item: dict[str, Any]) -> str:
        if any(str(i).startswith("unsafe_call") for i in item["issues"]):
            return "Review unsafe dynamic execution and replace with explicit, validated execution paths."
        if item["relative_path"] in {"main.py", "core/api.py"}:
            return "Reduce central registration risk with smaller registration modules and selftest coverage."
        if item["complexity_score"] >= 70:
            return "Split large module into smaller services and add targeted regression tests before any change."
        if item["risky_imports"]:
            return "Review risky imports and ensure operations remain gated by policy and explicit user approval."
        return "Review for maintainability and add regression coverage before refactoring."

    def _candidate_description(self, item: dict[str, Any]) -> str:
        return (
            f"Core file '{item['relative_path']}' should be reviewed. "
            f"Risk score={item['risk_score']}, complexity score={item['complexity_score']}, "
            f"issues={', '.join(item['issues']) or 'none'}. "
            "This is a review-only Core Evolution proposal and must not modify files automatically."
        )

    def _grade(self, score: int) -> str:
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "E"

    def _read_history(self) -> list[dict[str, Any]]:
        if not self.history_store.exists():
            return []
        try:
            data = json.loads(self.history_store.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _append_history(self, event: dict[str, Any]) -> None:
        self.history_store.parent.mkdir(parents=True, exist_ok=True)
        items = self._read_history()
        items.append({"timestamp": _now(), **event})
        self.history_store.write_text(json.dumps(items[-500:], indent=2, ensure_ascii=False), encoding="utf-8")
