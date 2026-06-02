from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIRS_TO_EMPTY = [
    ROOT / "logs",
    ROOT / "memory" / "chat_sessions",
    ROOT / "memory" / "task_plans",
    ROOT / "memory" / "task_executions",
    ROOT / "memory" / "reasoning",
    ROOT / "sandbox" / "runs",
    ROOT / "sandbox" / "tmp",
    ROOT / "tool_proposals",
    ROOT / "skill_proposals",
    ROOT / "proposals" / "improvements",
]

FILES_TO_RESET = {
    ROOT / "memory" / "chat_sessions.json": '{\n  "sessions": []\n}\n',
    ROOT / "memory" / "conversation_memory.json": '{\n  "facts": {}\n}\n',
    ROOT / "memory" / "coordinator_log.jsonl": "",
    ROOT / "memory" / "planner_agent_log.jsonl": "",
    ROOT / "memory" / "worker_agent_log.jsonl": "",
    ROOT / "memory" / "sandbox_log.jsonl": "",
    ROOT / "memory" / "conversation_memory_log.jsonl": "",
    ROOT / "memory" / "capability_event_log.jsonl": "",
    ROOT / "memory" / "capability_workflow_log.jsonl": "",
    ROOT / "memory" / "tool_generation_log.jsonl": "",
    ROOT / "memory" / "reality_check_log.jsonl": "",
}


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


def main() -> None:
    for pattern in ["__pycache__", ".pytest_cache"]:
        for path in ROOT.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
    for path in ROOT.rglob("*.pyc"):
        path.unlink()
    for path in DIRS_TO_EMPTY:
        empty_dir(path)
    for path, content in FILES_TO_RESET.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
