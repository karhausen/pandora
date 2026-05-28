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
