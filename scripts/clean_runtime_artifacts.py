from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Static configuration under config/ is intentionally not cleaned here.

DIRS_TO_EMPTY = [
    ROOT / "logs",
    ROOT / "memory" / "chat_sessions",
    ROOT / "memory" / "task_plans",
    ROOT / "memory" / "task_executions",
    ROOT / "memory" / "reasoning",
    ROOT / "sandbox" / "runs",
    ROOT / "tool_proposals",
    ROOT / "skill_proposals",
    ROOT / "proposals" / "improvements",
]

FILES_TO_RESET = {
    ROOT / "memory" / "chat_sessions.json": '{\n  "sessions": []\n}\n',
    ROOT / "memory" / "conversation_memory.json": '{\n  "facts": {}\n}\n',
    ROOT / "memory" / "tool_usage_stats.json": "{}\n",
}

FILES_TO_REMOVE = [
    ROOT / "memory" / "governance_report.json",
]




def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

def empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (path / ".gitkeep").touch()


def reset_tool_registry_to_base_tools() -> None:
    registry = ROOT / "config" / "tools" / "tool_registry.json"
    if not registry.exists():
        return
    data = json.loads(registry.read_text(encoding="utf-8"))
    base = {
        tool_id: meta
        for tool_id, meta in data.items()
        if str(meta.get("module", "")).startswith("tools.") and not meta.get("installed_from")
    }
    registry.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for runtime_path in [ROOT / "sandbox" / "tmp", ROOT / "sandbox" / "runs"]:
        remove_path(runtime_path)
    for pattern in ["__pycache__", ".pytest_cache"]:
        for path in ROOT.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
    for path in ROOT.rglob("*.pyc"):
        path.unlink()
    generated_tools = ROOT / "generated_tools"
    generated_tools.mkdir(parents=True, exist_ok=True)
    for child in generated_tools.iterdir():
        if child.name in {"__init__.py", ".gitkeep"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (generated_tools / "__init__.py").touch()
    for path in DIRS_TO_EMPTY:
        empty_dir(path)
    reset_tool_registry_to_base_tools()
    for path in ROOT.rglob("*.jsonl"):
        if "memory" in path.parts:
            remove_path(path)
    for path in FILES_TO_REMOVE:
        remove_path(path)
    for path, content in FILES_TO_RESET.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
