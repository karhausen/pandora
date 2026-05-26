from __future__ import annotations
import asyncio, json, sqlite3, uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from .config import TASK_RUNTIME_DB
from .models import RuntimeTask, TaskKind, TaskStatus
from .planner import Planner
from .skill_executor import SkillExecutor
from .skill_registry import SkillRegistry
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry

class TaskStore:
    def __init__(self, path: Path = TASK_RUNTIME_DB):
        self.path=path; self.path.parent.mkdir(parents=True, exist_ok=True); self._init_db()
    def _connect(self): return sqlite3.connect(self.path)
    def _init_db(self):
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, task TEXT, target TEXT, payload TEXT NOT NULL, auto_create INTEGER NOT NULL, priority INTEGER NOT NULL, result TEXT, error TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT)''')
    def create(self, kind: TaskKind, task=None, target=None, payload=None, auto_create=False, priority=5):
        rt=RuntimeTask(id=str(uuid.uuid4()), kind=kind, status=TaskStatus.QUEUED, task=task, target=target, payload=payload or {}, auto_create=auto_create, priority=priority, created_at=datetime.now(UTC).isoformat())
        with self._connect() as con:
            con.execute("INSERT INTO tasks(id,kind,status,task,target,payload,auto_create,priority,created_at) VALUES (?,?,?,?,?,?,?,?,?)", (rt.id, rt.kind.value, rt.status.value, rt.task, rt.target, json.dumps(rt.payload), int(rt.auto_create), rt.priority, rt.created_at))
        return rt
    def update(self, task_id, **fields):
        allowed={"status","result","error","started_at","finished_at"}; updates=[]; values=[]
        for k,v in fields.items():
            if k in allowed:
                updates.append(f"{k}=?"); values.append(json.dumps(v, ensure_ascii=False) if k=="result" and v is not None else v)
        if not updates: return
        values.append(task_id)
        with self._connect() as con: con.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", values)
    def get(self, task_id):
        with self._connect() as con:
            con.row_factory=sqlite3.Row; row=con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._row(row) if row else None
    def list(self, limit=50):
        with self._connect() as con:
            con.row_factory=sqlite3.Row; rows=con.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]
    def queued(self, limit=10):
        with self._connect() as con:
            con.row_factory=sqlite3.Row; rows=con.execute("SELECT * FROM tasks WHERE status=? ORDER BY priority ASC, created_at ASC LIMIT ?", (TaskStatus.QUEUED.value, limit)).fetchall()
        return [self._row(r) for r in rows]
    def cancel(self, task_id):
        t=self.get(task_id)
        if not t or t.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING}: return False
        self.update(task_id, status=TaskStatus.CANCELLED.value, finished_at=datetime.now(UTC).isoformat()); return True
    def _row(self,row):
        return RuntimeTask(id=row["id"], kind=TaskKind(row["kind"]), status=TaskStatus(row["status"]), task=row["task"], target=row["target"], payload=json.loads(row["payload"]), auto_create=bool(row["auto_create"]), priority=int(row["priority"]), result=json.loads(row["result"]) if row["result"] else None, error=row["error"], created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"])

class TaskRuntime:
    def __init__(self, store=None):
        self.store=store or TaskStore(); self._worker_task=None; self._stop=asyncio.Event()
    async def start(self):
        if self._worker_task is None or self._worker_task.done():
            self._stop.clear(); self._worker_task=asyncio.create_task(self._worker_loop())
    async def stop(self):
        self._stop.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    async def _worker_loop(self):
        while not self._stop.is_set():
            q=self.store.queued(1)
            if not q: await asyncio.sleep(0.1); continue
            await self.execute_task(q[0].id)
    async def execute_task(self, task_id):
        task=self.store.get(task_id)
        if not task: raise ValueError(f"Task not found: {task_id}")
        self.store.update(task_id,status=TaskStatus.RUNNING.value,started_at=datetime.now(UTC).isoformat())
        try:
            result=await self._dispatch(task)
            self.store.update(task_id,status=TaskStatus.COMPLETED.value,result=result,finished_at=datetime.now(UTC).isoformat())
        except Exception as exc:
            self.store.update(task_id,status=TaskStatus.FAILED.value,error=f"{type(exc).__name__}: {exc}",finished_at=datetime.now(UTC).isoformat())
        return self.store.get(task_id)
    async def _dispatch(self, task):
        if task.kind==TaskKind.ANALYZE: return Planner().analyze_task(task.task or "")
        if task.kind==TaskKind.ENSURE_CAPABILITY: return Planner().ensure_capabilities(task.task or "", auto_create=task.auto_create)
        if task.kind==TaskKind.TOOL:
            r=ToolRegistry(); r.discover(); return (await ToolExecutor(r).run_tool(task.target or "", task.payload, task=task.task)).model_dump()
        if task.kind==TaskKind.SKILL:
            tr=ToolRegistry(); tr.discover(); sr=SkillRegistry(); sr.discover(); return (await SkillExecutor(sr,tr).run_skill(task.target or "", task.payload, task=task.task)).model_dump()
        raise ValueError(f"Unsupported task kind: {task.kind}")
