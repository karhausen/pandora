from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoreConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    tool_dir: Path = project_root / "tools"
    skill_dir: Path = project_root / "skills"
    memory_dir: Path = project_root / "memory"
    log_dir: Path = project_root / "logs"
    heartbeat_timeout_ms: int = 1500
    tool_timeout_seconds: int = 10
    safe_mode: bool = False
    llm_provider: str = "stub"  # stub | ollama | openai
    ollama_model_small: str = "llama3.2:3b"
    ollama_model_strong: str = "qwen2.5-coder:7b"

    def ensure_dirs(self) -> None:
        for path in [self.tool_dir, self.skill_dir, self.memory_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)
