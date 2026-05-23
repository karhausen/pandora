from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agent_core import AgentCore


class TaskRequest(BaseModel):
    task: str


def create_app() -> FastAPI:
    app = FastAPI(title="Local Autonomous Agent", version="0.1.0")
    core = AgentCore()
    core.initialize()

    @app.get("/status")
    def status() -> dict:
        return core.status()

    @app.get("/heartbeat")
    def heartbeat() -> dict:
        return core.heartbeat.as_dict()

    @app.post("/task")
    def task(req: TaskRequest) -> dict:
        return core.run_task(req.task)

    @app.get("/tools")
    def tools() -> dict:
        return {"tools": core.registry.list_names()}

    @app.get("/skills")
    def skills() -> dict:
        return {"skills": []}

    @app.get("/memory/short-term")
    def short_term() -> dict:
        return core.memory.get_short_term_all()

    @app.get("/core-versions")
    def core_versions() -> dict:
        return {"active": "0.1.0", "versions": [], "note": "MVP7 erweitert dies."}

    @app.get("/improvement-proposals")
    def improvement_proposals() -> dict:
        return {"proposals": [], "note": "MVP5 erweitert dies."}

    return app


app = create_app()
