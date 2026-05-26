from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class SecurityLevel(str, Enum):
    SAFE = "SAFE"
    LIMITED = "LIMITED"
    DANGEROUS = "DANGEROUS"
    SYSTEM = "SYSTEM"


class ToolStatus(str, Enum):
    GENERATED = "GENERATED"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class SkillStatus(str, Enum):
    GENERATED = "GENERATED"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class ProposalStatus(str, Enum):
    DRAFT = "DRAFT"
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ACTIVATED = "ACTIVATED"


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskKind(str, Enum):
    ANALYZE = "analyze"
    TOOL = "tool"
    SKILL = "skill"
    ENSURE_CAPABILITY = "ensure_capability"


class CoreVersionStatus(str, Enum):
    CREATED = "CREATED"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    STABLE = "STABLE"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


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


class CapabilityAnalysis(BaseModel):
    task: str
    required_capabilities: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    available_skills: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    recommended_action: str = "direct"


class Episode(BaseModel):
    id: str
    task: str
    kind: str
    success: bool
    used_tools: list[str] = Field(default_factory=list)
    used_skills: list[str] = Field(default_factory=list)
    execution_time: float = 0.0
    error: str | None = None
    summary: str | None = None
    created_at: str


class RuntimeTask(BaseModel):
    id: str
    kind: TaskKind
    status: TaskStatus = TaskStatus.QUEUED
    task: str | None = None
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    auto_create: bool = False
    priority: int = 5
    result: Any = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class CoreVersionMeta(BaseModel):
    version_id: str
    build_id: str
    created_at: str
    status: CoreVersionStatus = CoreVersionStatus.CREATED
    source: str = "snapshot"
    path: str
    tests_passed: bool | None = None
    heartbeat_passed: bool | None = None
    smoke_tests_passed: bool | None = None
    error: str | None = None
    rollback_target: str | None = None
