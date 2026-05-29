from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class SecurityLevel(str, Enum):
    SAFE = "SAFE"
    LIMITED = "LIMITED"
    DANGEROUS = "DANGEROUS"
    SYSTEM = "SYSTEM"


class ToolStatus(str, Enum):
    ACTIVE = "ACTIVE"
    VALIDATED = "VALIDATED"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class SkillStatus(str, Enum):
    ACTIVE = "ACTIVE"
    VALIDATED = "VALIDATED"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class LLMProvider(str, Enum):
    MOCK = "mock"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class LLMTaskType(str, Enum):
    CHAT = "chat"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_GENERATION = "tool_generation"
    REFLECTION = "reflection"
    CORE_REVIEW = "core_review"


class ToolMeta(BaseModel):
    id: str
    name: str
    description: str
    version: str = "0.1.0"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.SAFE
    status: ToolStatus = ToolStatus.ACTIVE
    module: str
    function: str = "run"


class ToolResult(BaseModel):
    success: bool
    tool: str
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0


class SkillStep(BaseModel):
    id: str
    type: str = "tool"
    tool_id: str | None = None
    input_map: dict[str, str] = Field(default_factory=dict)
    static_input: dict[str, Any] = Field(default_factory=dict)
    save_as: str | None = None


class SkillMeta(BaseModel):
    id: str
    name: str
    description: str
    version: str = "0.1.0"
    status: SkillStatus = SkillStatus.ACTIVE
    security_level: SecurityLevel = SecurityLevel.SAFE
    required_tools: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    steps: list[SkillStep] = Field(default_factory=list)


class SkillResult(BaseModel):
    success: bool
    skill: str
    output: Any = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    execution_time: float = 0.0


class LLMRequest(BaseModel):
    task_type: LLMTaskType = LLMTaskType.CHAT
    prompt: str
    system_prompt: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    provider_name: str | None = None
    expect_json: bool = False
    timeout: float = 20.0


class LLMResponse(BaseModel):
    success: bool
    provider: LLMProvider
    provider_name: str | None = None
    model: str
    content: str
    parsed_json: Any = None
    error: str | None = None
    raw: Any = None


class LLMRouteDecision(BaseModel):
    task_type: LLMTaskType
    provider: LLMProvider
    provider_name: str
    model: str
    reason: str


class LLMTaskAnalysis(BaseModel):
    task: str
    summary: str
    intent: str
    complexity: str = "low"
    required_capabilities: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    suggested_skills: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    risk_level: str = "LOW"
    next_action: str = "answer"


class AgentActionType(str, Enum):
    ANSWER = "answer"
    TOOL = "tool"
    SKILL = "skill"
    REJECT = "reject"


class AgentAction(BaseModel):
    type: AgentActionType
    tool_id: str | None = None
    skill_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AgentRunResult(BaseModel):
    run_id: str
    task: str
    success: bool
    analysis: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    evaluation: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str
    execution_time: float = 0.0


class ToolProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ToolSpec(BaseModel):
    id: str
    name: str
    description: str
    capability: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.SAFE


class ToolProposal(BaseModel):
    id: str
    status: ToolProposalStatus = ToolProposalStatus.PROPOSED
    capability: str
    spec: ToolSpec
    created_at: str
    proposal_dir: str
    code_file: str
    test_file: str
    validation: dict[str, Any] = Field(default_factory=dict)
    risk: str = "LOW"


class ToolActivationResult(BaseModel):
    activated: bool
    proposal_id: str
    tool_id: str | None = None
    copied_to: str | None = None
    registered: bool = False
    tested: bool = False
    error: str | None = None


class CapabilityEvent(BaseModel):
    event_id: str
    task: str
    gap_detected: bool
    capability: str | None = None
    action: str
    proposal_id: str | None = None
    created_at: str
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilityWorkflowResult(BaseModel):
    workflow_id: str
    task: str
    success: bool
    mode: str
    proposal_created: bool = False
    proposal_id: str | None = None
    activated: bool = False
    activation: dict[str, Any] | None = None
    retry_result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str


class SkillProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class SkillProposal(BaseModel):
    id: str
    status: SkillProposalStatus = SkillProposalStatus.PROPOSED
    skill: SkillMeta
    created_at: str
    proposal_dir: str
    validation: dict[str, Any] = Field(default_factory=dict)
    source: str = "journal"


class SkillActivationResult(BaseModel):
    activated: bool
    proposal_id: str
    skill_id: str | None = None
    copied_to: str | None = None
    registered: bool = False
    tested: bool = False
    error: str | None = None


class LearningSummary(BaseModel):
    learned: bool
    entries_analyzed: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    rankings: dict[str, Any] = Field(default_factory=dict)
    failures: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    strategies: dict[str, Any] = Field(default_factory=dict)


class ExecutionPolicyName(str, Enum):
    TRUSTED = "trusted"
    RESTRICTED = "restricted"
    ISOLATED = "isolated"
    DANGEROUS = "dangerous"


class ExecutionPolicy(BaseModel):
    name: ExecutionPolicyName
    timeout: float = 5.0
    allow_network: bool = False
    allow_shell: bool = False
    allow_write: bool = False
    allowed_paths: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    success: bool
    tool_id: str
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0
    policy: str = "restricted"
    isolated: bool = True
    returncode: int | None = None


class ToolGenerationAttempt(BaseModel):
    attempt: int
    success: bool
    code_review: dict[str, Any] = Field(default_factory=dict)
    test_result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolGenerationResult(BaseModel):
    success: bool
    capability: str
    proposal_created: bool = False
    proposal_id: str | None = None
    attempts: list[ToolGenerationAttempt] = Field(default_factory=list)
    error: str | None = None


class CoreVersionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    STABLE = "STABLE"
    FAILED = "FAILED"
    ROLLBACK = "ROLLBACK"
    SAFE_MODE = "SAFE_MODE"


class CoreVersion(BaseModel):
    version_id: str
    created_at: str
    status: CoreVersionStatus = CoreVersionStatus.CANDIDATE
    path: str
    heartbeat_passed: bool = False
    smoke_passed: bool = False
    activated_at: str | None = None
    notes: str | None = None


class CoreSmokeResult(BaseModel):
    success: bool
    tests: int
    passed: int
    failed: int
    details: dict[str, Any] = Field(default_factory=dict)


class CoreStatus(BaseModel):
    active_version: str | None = None
    safe_mode: bool = False
    rollback_available: bool = False
    last_smoke: dict[str, Any] = Field(default_factory=dict)
    last_heartbeat: dict[str, Any] = Field(default_factory=dict)


class RealityCheckIteration(BaseModel):
    iteration: int
    heartbeat: dict[str, Any] = Field(default_factory=dict)
    smoke: dict[str, Any] = Field(default_factory=dict)
    success: bool = False


class RealityCheckResult(BaseModel):
    success: bool
    iterations: int
    passed: int
    failed: int
    results: list[RealityCheckIteration] = Field(default_factory=list)
    snapshot_summary: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    step_id: str
    title: str
    action_type: str = "answer"
    tool_id: str | None = None
    skill_id: str | None = None
    capability: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class TaskPlan(BaseModel):
    plan_id: str
    task: str
    created_at: str
    provider_name: str | None = None
    model: str | None = None
    complexity: str = "simple"
    summary: str
    steps: list[PlanStep] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    ready_for_execution: bool = True
    raw_analysis: dict[str, Any] = Field(default_factory=dict)


class WorkerStepResult(BaseModel):
    step_id: str
    success: bool
    action_type: str
    tool_id: str | None = None
    skill_id: str | None = None
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0


class TaskExecutionResult(BaseModel):
    execution_id: str
    plan_id: str | None = None
    task: str
    success: bool
    created_at: str
    steps: list[WorkerStepResult] = Field(default_factory=list)
    final_output: Any = None
    error: str | None = None
    execution_time: float = 0.0
