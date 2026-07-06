from __future__ import annotations

import ast
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

def main() -> None:
    files = sorted(CORE.rglob("*.py"))
    modules = {str(p.relative_to(ROOT).with_suffix("")) .replace("/", ".") for p in files}
    refs: dict[str, set[str]] = collections.defaultdict(set)
    scan_files = files + list((ROOT / "tests").rglob("*.py")) + [ROOT / "main.py"]
    for path in scan_files:
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        src = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("core."):
                        refs[alias.name].add(src)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("core"):
                refs[node.module].add(src)
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    if candidate in modules:
                        refs[candidate].add(src)
    unreferenced = sorted(m for m in modules if not refs.get(m) and not m.endswith(".__init__"))
    print(json.dumps({
        "core_python_files": len(files),
        "modules": len(modules),
        "statically_unreferenced": len(unreferenced),
        "unreferenced_sample": unreferenced[:100],
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
