from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "tools"
SKILLS_DIR = ROOT_DIR / "skills"
MEMORY_DIR = ROOT_DIR / "memory"
LOGS_DIR = ROOT_DIR / "logs"
PROMPTS_DIR = ROOT_DIR / "prompts"
PROPOSALS_DIR = ROOT_DIR / "proposals"

TOOL_REGISTRY_FILE = MEMORY_DIR / "tool_registry.json"
SKILL_REGISTRY_FILE = MEMORY_DIR / "skill_registry.json"
LLM_CONFIG_FILE = MEMORY_DIR / "llm_config.json"
AGENT_JOURNAL_FILE = MEMORY_DIR / "agent_journal.jsonl"
IMPROVEMENTS_DIR = PROPOSALS_DIR / "improvements"

PROTECTED_CORE_FILES = {
    "heartbeat.py", "rollback_manager.py", "recovery.py", "security.py",
    "activation_manager.py", "version_manager.py", "config.py",
}

GENERATED_TOOLS_DIR = ROOT_DIR / "generated_tools"
TOOL_PROPOSALS_DIR = ROOT_DIR / "tool_proposals"
TOOL_SANDBOX_DIR = ROOT_DIR / "sandbox" / "tool_generation"
TOOL_ACTIVATION_LOG = MEMORY_DIR / "tool_activation_log.jsonl"

CAPABILITY_EVENT_LOG = MEMORY_DIR / "capability_events.jsonl"

CAPABILITY_WORKFLOW_LOG = MEMORY_DIR / "capability_workflow_log.jsonl"

SKILL_PROPOSALS_DIR = ROOT_DIR / "skill_proposals"
SKILL_ACTIVATION_LOG = MEMORY_DIR / "skill_activation_log.jsonl"
LEARNING_EVENTS_FILE = MEMORY_DIR / "learning_events.jsonl"
STRATEGY_MEMORY_FILE = MEMORY_DIR / "strategy_memory.json"
RANKINGS_FILE = MEMORY_DIR / "tool_skill_rankings.json"
FAILURE_ANALYSIS_FILE = MEMORY_DIR / "failure_analysis.json"
RECOMMENDATIONS_FILE = MEMORY_DIR / "recommendations.json"
DOCS_DIR = ROOT_DIR / "docs"
CHANGELOG_FILE = DOCS_DIR / "changelog.md"
GOVERNANCE_REPORT_FILE = MEMORY_DIR / "governance_report.json"
ARCHITECTURE_REPORT_FILE = MEMORY_DIR / "architecture_report.json"