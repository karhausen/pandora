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
EXCLUDE_RELATIVE = {".env", "config/llm/llm_config.local.json"}
RUNTIME_DIRS_TO_KEEP_EMPTY = {
    "logs",
    "memory/chat_sessions",
    "memory/task_plans",
    "memory/task_executions",
    "memory/reasoning",
    "sandbox/runs",
    "sandbox/tmp",
    "proposals/improvements",
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
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_mvp21_1_1_release_packaging.py", "-q"], cwd=dst, check=True)


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
    parser.add_argument("--version", default="mvp-21.1.1-release-packaging")
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
