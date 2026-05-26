from __future__ import annotations
from .capability_analyzer import CapabilityAnalyzer
from .reflection import ReflectionLogger
from .skill_registry import SkillRegistry
from .tool_registry import ToolRegistry

class Planner:
    def __init__(self, registry=None, skill_registry=None):
        self.registry=registry or ToolRegistry(); self.skill_registry=skill_registry or SkillRegistry(); self.analyzer=CapabilityAnalyzer(self.registry,self.skill_registry); self.reflection=ReflectionLogger()
    def analyze_task(self, task):
        self.registry.discover(); self.skill_registry.discover(); return self.analyzer.analyze(task).model_dump()
    def ensure_capabilities(self, task, auto_create=False):
        analysis=self.analyzer.analyze(task)
        self.reflection.record({"type":"capability_analysis","task":task,"missing_capabilities":analysis.missing_capabilities,"auto_create":auto_create})
        return {"analysis":analysis.model_dump(),"created_tools":[],"errors":[]}
