from __future__ import annotations
from .models import CapabilityAnalysis
from .tool_registry import ToolRegistry
from .skill_registry import SkillRegistry
KEYWORD_CAPABILITIES={"csv":"csv_processing","calculate":"calculation","rechne":"calculation","workflow":"workflow","skill":"workflow"}
class CapabilityAnalyzer:
    def __init__(self, registry: ToolRegistry, skill_registry: SkillRegistry | None = None):
        self.registry=registry; self.skill_registry=skill_registry or SkillRegistry()
    def analyze(self, task):
        task_l=task.lower(); required=sorted({cap for key,cap in KEYWORD_CAPABILITIES.items() if key in task_l})
        tools=[t.id for t in self.registry.list()]; skills=[s.id for s in self.skill_registry.list()]
        caps=set()
        for t in self.registry.list():
            txt=f"{t.name} {t.description} {t.id}".lower()
            if "calculator" in txt: caps.add("calculation")
            if "csv" in txt: caps.add("csv_processing")
        if skills: caps.add("workflow")
        missing=[c for c in required if c not in caps]
        return CapabilityAnalysis(task=task, required_capabilities=required, available_tools=tools, available_skills=skills, missing_capabilities=missing, recommended_action="create_tool" if missing else "direct_or_tool_or_skill")
