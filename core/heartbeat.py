from __future__ import annotations
import asyncio, time
from .episodic_memory import EpisodicMemory
from .memory import Memory
from .skill_quality import SkillQualityDB
from .skill_registry import SkillRegistry
from .task_runtime import TaskStore
from .tool_registry import ToolRegistry
from .tool_runtime import ToolRuntimeDB

class Heartbeat:
    def __init__(self):
        self.memory=Memory(); self.registry=ToolRegistry(); self.skill_registry=SkillRegistry(); self.runtime=ToolRuntimeDB(); self.episodic=EpisodicMemory(); self.skill_quality=SkillQualityDB(); self.task_store=TaskStore()
    async def check(self)->dict:
        start=time.perf_counter()
        status={"healthy":True,"planner":"ok","memory":"unknown","tool_registry":"unknown","skill_registry":"unknown","tool_runtime_db":"unknown","episodic_memory":"unknown","skill_quality_db":"unknown","task_runtime_db":"unknown","event_loop":"unknown","response_time":None}
        checks=[("memory",self.memory.get_all),("tool_registry",self.registry.list),("skill_registry",self.skill_registry.list),("tool_runtime_db",self.runtime.stats),("episodic_memory",lambda:self.episodic.list_recent(1)),("skill_quality_db",self.skill_quality.list),("task_runtime_db",lambda:self.task_store.list(1))]
        for name,fn in checks:
            try: fn(); status[name]="ok"
            except Exception as exc: status["healthy"]=False; status[name]=f"error: {exc}"
        try: await asyncio.sleep(0); status["event_loop"]="ok"
        except Exception as exc: status["healthy"]=False; status["event_loop"]=f"error: {exc}"
        status["response_time"]=round(time.perf_counter()-start,6)
        return status
