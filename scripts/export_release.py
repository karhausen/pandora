from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "htmlcov",
}

EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".log"}
EXCLUDE_FILE_NAMES = {".coverage", "coverage.xml", ".DS_Store", "Thumbs.db"}
EXCLUDE_RELATIVE = {".env", "config/llm/llm_config.local.json", "memory/maintenance.lock"}
RUNTIME_DIRS_TO_KEEP_EMPTY = {
    "logs",
    "memory/chat_sessions",
    "memory/task_plans",
    "memory/task_executions",
    "memory/reasoning",
    "sandbox/runs",
    "sandbox/tmp",
    "proposals/improvements",
    "proposals/maintenance_reports",
    "proposals/nightly_reviews",
    "proposals/tool_improvements",
    "proposals/capability_gaps",
    "proposals/capability_actions",
    "proposals/obsidian_import_candidates",
    "proposals/obsidian_import_executions",
    "proposals/review_inbox",
    "skill_proposals",
    "tool_proposals",
}

RESET_FILES = {
    "memory/chat_sessions.json": '{\n  "sessions": []\n}\n',
    "memory/conversation_memory.json": '{\n  "facts": {}\n}\n',
    "memory/tool_usage_stats.json": "{}\n",
    "memory/governance_report.json": '{}\n',
    "memory/coordinator_log.jsonl": "",
    "memory/planner_agent_log.jsonl": "",
    "memory/worker_agent_log.jsonl": "",
    "memory/sandbox_log.jsonl": "",
    "memory/conversation_memory_log.jsonl": "",
    "memory/capability_event_log.jsonl": "",
    "memory/capability_workflow_log.jsonl": "",
    "memory/tool_generation_log.jsonl": "",
    "memory/tool_lifecycle_log.jsonl": "",
    "memory/reality_check_log.jsonl": "",
    "memory/core_events.jsonl": "",
    "memory/maintenance_events.jsonl": "",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_exclude(path: Path, src: Path) -> bool:
    rel = path.relative_to(src).as_posix()
    parts = path.relative_to(src).parts
    if rel.startswith("release/build/") or rel == "release/build":
        return True
    if rel.startswith("release/") and path.suffix == ".zip":
        return True
    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return True
    if rel in EXCLUDE_RELATIVE:
        return True
    if path.name.startswith(".env.") and path.name != ".env.example":
        return True
    if path.name.endswith(".local.json") and not path.name.endswith(".local.example.json"):
        return True
    if path.name in EXCLUDE_FILE_NAMES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if ".egg-info" in parts:
        return True
    return False


def copy_source(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if should_exclude(path, src):
            continue
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def sanitize_tree(dst: Path) -> None:
    for blocked in EXCLUDE_DIR_NAMES:
        for path in list(dst.rglob(blocked)):
            if path.is_dir():
                shutil.rmtree(path)
    for path in list(dst.rglob("*.pyc")) + list(dst.rglob("*.pyo")) + list(dst.rglob("*.pyd")):
        if path.exists():
            path.unlink()
    for path in list(dst.rglob("*.local.json")):
        if not path.name.endswith(".local.example.json"):
            path.unlink()
    for env_file in list(dst.glob(".env.*")):
        if env_file.name != ".env.example":
            env_file.unlink()
    if (dst / ".env").exists():
        (dst / ".env").unlink()
    for relative in RUNTIME_DIRS_TO_KEEP_EMPTY:
        directory = dst / relative
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()
    for relative, content in RESET_FILES.items():
        path = dst / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    knowledge = dst / "user_knowledge"
    if knowledge.exists():
        shutil.rmtree(knowledge)
    for relative in (
        "user_knowledge/public",
        "user_knowledge/restricted_cloud_allowed",
        "user_knowledge/private_local_only",
    ):
        directory = dst / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()
    (dst / "user_knowledge" / "README.md").write_text(
        "# Pandora User Knowledge Base\n\n"
        "Lege hier eigene Markdown-, Text- oder JSON-Dateien ab.\n\n"
        "- `public/`: lokal + Cloud erlaubt.\n"
        "- `restricted_cloud_allowed/`: Cloud nach Policy-Prüfung.\n"
        "- `private_local_only/`: nur lokales LLM, niemals Cloud.\n",
        encoding="utf-8",
    )
    generated = dst / "generated_tools"
    generated.mkdir(exist_ok=True)
    for child in list(generated.iterdir()):
        if child.name in {"__init__.py", ".gitkeep"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (generated / "__init__.py").touch()


def run_tests(dst: Path) -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_mvp21_1_1_release_packaging.py", "tests/test_mvp21_2_maintenance_manager.py", "tests/test_mvp21_3_skill_candidate_pipeline.py", "tests/test_mvp21_4_tool_improvement_pipeline.py", "tests/test_mvp21_5_capability_gap_pipeline.py", "tests/test_mvp21_6_proposal_review_inbox.py", "tests/test_mvp21_7_proposal_approval_workflow.py", "tests/test_mvp21_8_gui_approval_api.py", "tests/test_mvp21_9_minimal_web_gui.py", "tests/test_mvp22_0_operations_dashboard.py", "tests/test_mvp22_1_user_gui_navigation.py", "tests/test_mvp22_2_tool_center_gui.py", "tests/test_mvp22_3_skill_center_gui.py", "tests/test_mvp22_4_memory_explorer.py", "tests/test_mvp22_5_night_mode_dashboard.py", "tests/test_mvp22_6_llm_profile_center.py", "tests/test_mvp22_6_1_llm_routing_editor.py", "tests/test_mvp22_6_2_user_gui_routing_sync.py", "tests/test_mvp22_7_user_knowledge_base.py", "tests/test_mvp22_8_knowledge_context_injection.py", "tests/test_mvp22_9_knowledge_metadata_governance.py", "tests/test_mvp22_9_1_knowledge_governance_hardening.py", "tests/test_mvp22_9_2_llm_fallback_diagnostics.py", "tests/test_mvp22_10_knowledge_editor_gui.py", "-q"], cwd=dst, check=True)


def run_audit(dst: Path) -> dict[str, object]:
    from scripts.release_audit import audit
    result = audit(dst)
    if not result["ok"]:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(2)
    return result


def make_manifest(dst: Path, version: str, audit_result: dict[str, object]) -> dict[str, object]:
    files = []
    for path in sorted(p for p in dst.rglob("*") if p.is_file()):
        rel = path.relative_to(dst).as_posix()
        if rel == "release_manifest.json":
            continue
        files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "project": "Pandora Agent",
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "sanitized": True,
        "audit_ok": bool(audit_result["ok"]),
        "file_count": len(files),
        "files": files,
    }
    (dst / "release_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def zip_tree(dst: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in dst.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(dst.parent).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a sanitized Pandora release ZIP.")
    parser.add_argument("--version", default="mvp-23.3.1-capability-actions-integration")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    build_root = ROOT / "release" / "build"
    package_dir = build_root / f"pandora_agent_{args.version.replace('-', '_').replace('.', '_')}"
    zip_path = Path(args.output) if args.output else ROOT / "release" / f"pandora_agent_{args.version.replace('-', '_').replace('.', '_')}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    copy_source(ROOT, package_dir)
    sanitize_tree(package_dir)
    if not args.skip_tests:
        run_tests(package_dir)
        sanitize_tree(package_dir)
    audit_result = run_audit(package_dir)
    make_manifest(package_dir, args.version, audit_result)
    audit_result = run_audit(package_dir)
    zip_tree(package_dir, zip_path)
    print(json.dumps({"ok": True, "zip": str(zip_path), "package_dir": str(package_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
