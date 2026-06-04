from __future__ import annotations

from pathlib import Path

from .config import (
    CONFIG_DIR,
    LLM_CONFIG_DIR,
    TOOLS_CONFIG_DIR,
    SKILLS_CONFIG_DIR,
    SYSTEM_CONFIG_DIR,
    LLM_CONFIG_FILE,
    LLM_CONFIG_TEMPLATE_FILE,
    LLM_CONFIG_LOCAL_FILE,
    TOOL_REGISTRY_FILE,
    SKILL_REGISTRY_FILE,
    EXECUTION_POLICY_FILE,
    SYSTEM_CONFIG_FILE,
)


class ConfigManager:
    """Central place for static Pandora configuration paths.

    Runtime state stays in memory/. Static configuration lives in config/.
    """

    def __init__(self, root: Path = CONFIG_DIR):
        self.root = root

    def ensure_dirs(self) -> None:
        for path in [CONFIG_DIR, LLM_CONFIG_DIR, TOOLS_CONFIG_DIR, SKILLS_CONFIG_DIR, SYSTEM_CONFIG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def llm_config(self) -> Path:
        return LLM_CONFIG_FILE

    @property
    def llm_template(self) -> Path:
        return LLM_CONFIG_TEMPLATE_FILE

    @property
    def llm_local(self) -> Path:
        return LLM_CONFIG_LOCAL_FILE

    @property
    def tool_registry(self) -> Path:
        return TOOL_REGISTRY_FILE

    @property
    def skill_registry(self) -> Path:
        return SKILL_REGISTRY_FILE

    @property
    def execution_policy(self) -> Path:
        return EXECUTION_POLICY_FILE

    @property
    def system_config(self) -> Path:
        return SYSTEM_CONFIG_FILE

    def summary(self) -> dict:
        self.ensure_dirs()
        return {
            "config_dir": str(CONFIG_DIR),
            "llm": {
                "config": str(LLM_CONFIG_FILE),
                "template": str(LLM_CONFIG_TEMPLATE_FILE),
                "local": str(LLM_CONFIG_LOCAL_FILE),
                "local_exists": LLM_CONFIG_LOCAL_FILE.exists(),
            },
            "tools": {
                "registry": str(TOOL_REGISTRY_FILE),
                "execution_policy": str(EXECUTION_POLICY_FILE),
            },
            "skills": {"registry": str(SKILL_REGISTRY_FILE)},
            "system": {"config": str(SYSTEM_CONFIG_FILE)},
        }
