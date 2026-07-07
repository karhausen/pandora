#!/usr/bin/env python3
"""MVP 30.9 ANALYZE tool: build a static runtime/import map for Pandora core.

This script performs *static* analysis only. It does not modify core files and it
must not be treated as proof that a module is safe to delete. Dynamic imports,
CLI-only code paths, plugin loading and string-based imports may be missed.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTRYPOINTS = ["main", "core.api"]
REPORT_DIR = PROJECT_ROOT / "docs"
JSON_REPORT = REPORT_DIR / "core_runtime_analysis_mvp30_9.json"
MD_REPORT = REPORT_DIR / "core_runtime_analysis_mvp30_9.md"


@dataclass
class ModuleInfo:
    module: str
    path: str
    imports_core: list[str]
    imported_by: list[str]
    reachable_from_entrypoints: bool
    defines_classes: list[str]
    defines_functions: list[str]
    legacy_candidate: bool
    notes: list[str]


def module_name_for(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def file_for_module(module: str) -> Path:
    return PROJECT_ROOT / Path(*module.split(".")).with_suffix(".py")


def iter_python_files() -> Iterable[Path]:
    excluded = {".venv", "__pycache__", ".pytest_cache", ".git"}
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        yield path


def resolve_imported_name(name: str, known_modules: set[str]) -> str | None:
    """Resolve an import to the most specific known module prefix.

    Example: core.chat_service.ChatService -> core.chat_service if that module
    exists. Returns None for imports outside the project/core graph.
    """
    if not name:
        return None
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in known_modules:
            return candidate
    return None


def parse_module(path: Path, known_modules: set[str]) -> tuple[list[str], list[str], list[str]]:
    imports: set[str] = set()
    classes: list[str] = []
    functions: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return [], [], []

    current_module = module_name_for(path)
    current_pkg_parts = current_module.split(".")[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = resolve_imported_name(alias.name, known_modules)
                if resolved:
                    imports.add(resolved)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                # Resolve relative imports conservatively.
                pkg = current_pkg_parts[: max(0, len(current_pkg_parts) - node.level + 1)]
                if base:
                    base = ".".join(pkg + base.split("."))
                else:
                    base = ".".join(pkg)
            # from core.foo import Bar -> core.foo
            resolved_base = resolve_imported_name(base, known_modules)
            if resolved_base:
                imports.add(resolved_base)
            # from core import chat_service -> core.chat_service
            for alias in node.names:
                if alias.name == "*":
                    continue
                full = f"{base}.{alias.name}" if base else alias.name
                resolved_full = resolve_imported_name(full, known_modules)
                if resolved_full:
                    imports.add(resolved_full)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

    imports.discard(current_module)
    return sorted(imports), sorted(classes), sorted(functions)


def reachable_modules(import_graph: dict[str, list[str]], entrypoints: list[str]) -> set[str]:
    seen: set[str] = set()
    q = deque([e for e in entrypoints if e in import_graph])
    while q:
        mod = q.popleft()
        if mod in seen:
            continue
        seen.add(mod)
        for dep in import_graph.get(mod, []):
            if dep not in seen:
                q.append(dep)
    return seen


def analyze() -> dict:
    py_files = sorted(iter_python_files())
    known_modules = {module_name_for(p) for p in py_files}
    core_modules = {m for m in known_modules if m == "core" or m.startswith("core.")}

    imports_by_module: dict[str, list[str]] = {}
    class_defs: dict[str, list[str]] = {}
    function_defs: dict[str, list[str]] = {}
    for path in py_files:
        mod = module_name_for(path)
        imports, classes, functions = parse_module(path, known_modules)
        # Keep only core/project imports. Main/tests/scripts can import core too.
        imports_by_module[mod] = [i for i in imports if i in known_modules]
        class_defs[mod] = classes
        function_defs[mod] = functions

    reverse: dict[str, list[str]] = defaultdict(list)
    for mod, deps in imports_by_module.items():
        for dep in deps:
            reverse[dep].append(mod)
    for mod in known_modules:
        reverse[mod] = sorted(set(reverse.get(mod, [])))

    entrypoints = [e for e in DEFAULT_ENTRYPOINTS if e in known_modules]
    reachable = reachable_modules(imports_by_module, entrypoints)

    module_infos: list[ModuleInfo] = []
    for mod in sorted(core_modules):
        path = str(file_for_module(mod).relative_to(PROJECT_ROOT))
        imported_by = reverse.get(mod, [])
        notes: list[str] = []
        is_reachable = mod in reachable
        legacy_candidate = False
        if not imported_by and mod not in entrypoints and mod != "core.__init__":
            notes.append("not_imported_by_static_graph")
        if not is_reachable:
            notes.append("not_reachable_from_main_or_core_api_static_graph")
        if (
            not imported_by
            and not is_reachable
            and mod not in entrypoints
            and not mod.endswith(".__init__")
            and mod != "core"
        ):
            legacy_candidate = True
            notes.append("legacy_candidate_static_only")
        module_infos.append(ModuleInfo(
            module=mod,
            path=path,
            imports_core=[d for d in imports_by_module.get(mod, []) if d in core_modules],
            imported_by=imported_by,
            reachable_from_entrypoints=is_reachable,
            defines_classes=class_defs.get(mod, []),
            defines_functions=function_defs.get(mod, []),
            legacy_candidate=legacy_candidate,
            notes=notes,
        ))

    return {
        "kind": "mvp30_9_static_core_runtime_analysis",
        "rules": [
            "ANALYZE-MVP: no core runtime files are modified by this report.",
            "Static analysis only: do not delete or move files based solely on this report.",
            "Router must remain dispatcher-only; this report does not re-enable tools/capability gap/evolution.",
        ],
        "entrypoints": entrypoints,
        "counts": {
            "python_files_total": len(py_files),
            "core_modules_total": len(core_modules),
            "reachable_core_modules": sum(1 for m in core_modules if m in reachable),
            "not_reachable_core_modules": sum(1 for m in core_modules if m not in reachable),
            "legacy_candidates_static_only": sum(1 for i in module_infos if i.legacy_candidate),
        },
        "modules": [asdict(i) for i in module_infos],
        "legacy_candidates_static_only": [asdict(i) for i in module_infos if i.legacy_candidate],
        "not_reachable_from_entrypoints": [asdict(i) for i in module_infos if not i.reachable_from_entrypoints],
    }


def write_markdown(report: dict) -> str:
    counts = report["counts"]
    legacy = report["legacy_candidates_static_only"]
    not_reachable = report["not_reachable_from_entrypoints"]
    lines: list[str] = []
    lines.append("# MVP 30.9 – Core Runtime Analysis")
    lines.append("")
    lines.append("Status: **ANALYZE-MVP**. Dieser Bericht verändert keine Core-Runtime-Dateien.")
    lines.append("")
    lines.append("## Regeln")
    lines.append("")
    for rule in report["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("## Entry Points")
    lines.append("")
    for ep in report["entrypoints"]:
        lines.append(f"- `{ep}`")
    lines.append("")
    lines.append("## Ergebnisübersicht")
    lines.append("")
    lines.append(f"- Python-Dateien gesamt: **{counts['python_files_total']}**")
    lines.append(f"- Core-Module gesamt: **{counts['core_modules_total']}**")
    lines.append(f"- Von Entry Points statisch erreichbar: **{counts['reachable_core_modules']}**")
    lines.append(f"- Nicht von Entry Points statisch erreichbar: **{counts['not_reachable_core_modules']}**")
    lines.append(f"- Legacy-Kandidaten, statisch/konservativ: **{counts['legacy_candidates_static_only']}**")
    lines.append("")
    lines.append("## Wichtige Einschränkung")
    lines.append("")
    lines.append("Diese Analyse ist statisch. Sie erkennt normale `import`/`from ... import ...`-Beziehungen, aber keine dynamischen Imports, CLI-Pfade, Plugin-Loader oder String-basierte Modulaufrufe. Ein Legacy-Kandidat darf deshalb **nicht automatisch gelöscht** werden.")
    lines.append("")
    lines.append("## Legacy-Kandidaten – statisch, nicht importiert und nicht erreichbar")
    lines.append("")
    if legacy:
        for item in legacy[:80]:
            lines.append(f"- `{item['path']}`")
        if len(legacy) > 80:
            lines.append(f"- … {len(legacy) - 80} weitere, siehe JSON-Bericht")
    else:
        lines.append("Keine eindeutigen Kandidaten gefunden.")
    lines.append("")
    lines.append("## Nicht erreichbare Core-Module – statischer Graph")
    lines.append("")
    for item in not_reachable[:120]:
        imported_by = ", ".join(f"`{x}`" for x in item["imported_by"][:5]) or "—"
        lines.append(f"- `{item['path']}` — imported_by: {imported_by}")
    if len(not_reachable) > 120:
        lines.append(f"- … {len(not_reachable) - 120} weitere, siehe JSON-Bericht")
    lines.append("")
    lines.append("## Nächste sinnvolle Aktion")
    lines.append("")
    lines.append("1. Bericht prüfen.")
    lines.append("2. Legacy-Kandidaten manuell klassifizieren: `active`, `cli_only`, `api_only`, `deprecated`, `unknown`.")
    lines.append("3. Erst danach Dateien nach `legacy/` verschieben.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    REPORT_DIR.mkdir(exist_ok=True)
    report = analyze()
    JSON_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    MD_REPORT.write_text(write_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MD_REPORT.relative_to(PROJECT_ROOT)}")
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
