from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "tools"
SKILLS_DIR = ROOT_DIR / "skills"
MEMORY_DIR = ROOT_DIR / "memory"
LOGS_DIR = ROOT_DIR / "logs"
PROPOSALS_DIR = ROOT_DIR / "proposals"
CORE_VERSIONS_DIR = ROOT_DIR / "core_versions"
CORE_VERSION_STORE = CORE_VERSIONS_DIR / "versions"
CORE_VERSION_MANIFEST = CORE_VERSIONS_DIR / "manifest.json"
ACTIVE_VERSION_FILE = CORE_VERSIONS_DIR / "active_version.txt"
STABLE_VERSION_FILE = CORE_VERSIONS_DIR / "stable_version.txt"

TOOL_REGISTRY_FILE = MEMORY_DIR / "tool_registry.json"
SKILL_REGISTRY_FILE = MEMORY_DIR / "skill_registry.json"
TOOL_RUNTIME_DB = MEMORY_DIR / "tool_runtime.sqlite"
SHORT_TERM_MEMORY = MEMORY_DIR / "short_term.json"
REFLECTION_LOG = MEMORY_DIR / "reflections.jsonl"
EPISODIC_DB = MEMORY_DIR / "episodic.sqlite"
SKILL_QUALITY_DB = MEMORY_DIR / "skill_quality.sqlite"
TASK_RUNTIME_DB = MEMORY_DIR / "task_runtime.sqlite"

PROTECTED_CORE_FILES = {
    "heartbeat.py",
    "rollback_manager.py",
    "recovery.py",
    "security.py",
    "activation_manager.py",
    "version_manager.py",
    "config.py",
}
