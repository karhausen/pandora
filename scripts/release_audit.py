from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_BLOCKED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".venv",
    "venv",
    "dist",
    "build",
    "htmlcov",
}

RUNTIME_DIRS = {
    "logs",
    "runtime",
    "tmp",
    "temp",
    "test_results",
    "sandbox/runs",
    "sandbox/tmp",
    "core_versions/runtime",
    "memory/checkpoints",
    "memory/snapshots",
}

BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
}

BLOCKED_FILENAMES = {
    ".coverage",
    "coverage.xml",
    ".DS_Store",
    "Thumbs.db",
}

BLOCKED_EXACT_RELATIVE = {
    ".env",
    "config/llm/llm_config.local.json",
}

SECRET_PATTERNS = [
    re.compile(r"sk-(?!testsecret)[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(?<![_-])(api[_-]?key|secret|token|password)\s*[:=]\s*['\"](?!env:|your_|OPENAI_|COMPANY_|LOCAL_|PLACEHOLDER|example|lm-studio)[^'\"\s]{12,}['\"]"),
]

ALLOW_SECRET_FILES = {
    ".env.example",
    "config/llm/llm_config.local.example.json",
    "config/llm/llm_config.template.json",
    "docs/release_packaging.md",
}

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".example", ".dockerignore", ".gitignore"
}


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "path": self.path, "message": self.message}


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_text_candidate(path: Path) -> bool:
    if path.name in {".gitignore", ".dockerignore"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    issues: list[AuditIssue] = []

    for path in root.rglob("*"):
        if path == root:
            continue
        relative = rel(path, root)
        parts = set(Path(relative).parts)
        if path.is_dir() and path.name in DEFAULT_BLOCKED_DIRS:
            issues.append(AuditIssue("error", relative, "blocked cache/build/venv directory must not be released"))
        if path.is_dir() and relative in RUNTIME_DIRS:
            children = [p.name for p in path.iterdir() if p.name != ".gitkeep"]
            if children:
                issues.append(AuditIssue("error", relative, "runtime directory contains release-blocked content"))
        if path.is_file():
            if path.suffix in BLOCKED_SUFFIXES:
                issues.append(AuditIssue("error", relative, "blocked runtime/binary file suffix"))
            if path.name in BLOCKED_FILENAMES:
                issues.append(AuditIssue("error", relative, "blocked generated/local file"))
            if relative in BLOCKED_EXACT_RELATIVE or path.name.startswith(".env.") and path.name != ".env.example":
                issues.append(AuditIssue("error", relative, "local secret/config file must not be released"))
            if ".egg-info" in parts:
                issues.append(AuditIssue("error", relative, "build metadata must not be released"))
            if path.name.endswith(".local.json") and not path.name.endswith(".local.example.json"):
                issues.append(AuditIssue("error", relative, "local configuration file must not be released"))

    for path in iter_files(root):
        relative = rel(path, root)
        if relative in ALLOW_SECRET_FILES or not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(AuditIssue("error", relative, "possible inline secret detected"))
                break

    errors = [i for i in issues if i.severity == "error"]
    return {
        "ok": not errors,
        "root": str(root),
        "issue_count": len(issues),
        "issues": [i.as_dict() for i in issues],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Pandora release tree before packaging.")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(Path(args.root))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Pandora Release Audit")
        print(f"OK: {result['ok']}")
        for issue in result["issues"]:
            print(f"[{issue['severity'].upper()}] {issue['path']}: {issue['message']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
