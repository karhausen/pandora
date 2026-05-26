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
    success_count: int = 0
    failure_count: int = 0
    last_error: str | None = None


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


class ToolSpec(BaseModel):
    id: str
    name: str
    description: str
    capability: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    code: str
    tests: list[dict[str, Any]] = Field(default_factory=list)
