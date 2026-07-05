from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "data" / "execution_trace"
TRACE_FILE = TRACE_DIR / "events.jsonl"
STATE_FILE = TRACE_DIR / "state.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


COMPONENTS = [
    "local_llm",
    "cloud_llm",
    "python",
    "tool",
    "knowledge",
    "memory",
    "evolution",
    "proposal",
]

STATES = {"idle", "active", "waiting", "success", "warning", "error", "skipped"}


@dataclass
class TraceEvent:
    event_id: str
    trace_id: str
    timestamp: str
    component: str
    state: str
    title: str
    detail: str | None = None
    duration_ms: float | None = None
    provider_name: str | None = None
    model: str | None = None
    route: str | None = None
    source: str = "python"
    payload: dict[str, Any] | None = None


class ExecutionTraceManager:
    """Small persistent execution trace for the user chat sidebar.

    This service is intentionally observational only. It never changes routing,
    never approves proposals, and never activates tools. It records which major
    components were involved in a request so bypasses are visible in the GUI.
    """

    VERSION = "29.7.3"

    def __init__(self) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        state = self.current_state()
        return {
            "kind": "execution_trace_status",
            "version": self.VERSION,
            "ok": True,
            "enabled": True,
            "mode": "observability_only",
            "components": COMPONENTS,
            "current_trace_id": state.get("trace_id"),
            "event_count": len(self.events(limit=10000)),
            "activates_changes": False,
            "generated_at": _utc_now(),
        }

    def start(self, task: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        state = {
            "trace_id": trace_id,
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "task": task,
            "session_id": session_id,
            "components": {name: {"state": "idle", "label": self._label(name), "last_event": None} for name in COMPONENTS},
            "current_step": "Request received",
            "finished": False,
        }
        self._write_state(state)
        self.record(trace_id, "python", "active", "Request received", detail=task, source="user_gui")
        return self.current_state()

    def record(
        self,
        trace_id: str | None,
        component: str,
        state: str,
        title: str,
        detail: str | None = None,
        duration_ms: float | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        route: str | None = None,
        source: str = "python",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if component not in COMPONENTS:
            component = "python"
        if state not in STATES:
            state = "warning"
        if not trace_id:
            trace_id = self.current_state().get("trace_id") or f"trace_{uuid.uuid4().hex[:12]}"
        event = TraceEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            trace_id=trace_id,
            timestamp=_utc_now(),
            component=component,
            state=state,
            title=title,
            detail=detail,
            duration_ms=duration_ms,
            provider_name=provider_name,
            model=model,
            route=route,
            source=source,
            payload=payload or {},
        )
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False, default=str) + "\n")
        state_doc = self.current_state()
        if not state_doc.get("trace_id") or state_doc.get("trace_id") != trace_id:
            state_doc = {
                "trace_id": trace_id,
                "started_at": event.timestamp,
                "updated_at": event.timestamp,
                "task": None,
                "session_id": None,
                "components": {name: {"state": "idle", "label": self._label(name), "last_event": None} for name in COMPONENTS},
                "current_step": title,
                "finished": False,
            }
        state_doc["updated_at"] = event.timestamp
        state_doc["current_step"] = title
        comp = state_doc.setdefault("components", {}).setdefault(component, {"label": self._label(component)})
        comp.update({
            "state": state,
            "label": self._label(component),
            "last_event": title,
            "detail": detail,
            "provider_name": provider_name,
            "model": model,
            "route": route,
            "updated_at": event.timestamp,
        })
        if state in {"success", "error"} and title.lower().startswith("request"):
            state_doc["finished"] = True
        self._write_state(state_doc)
        return asdict(event)

    def finish(self, trace_id: str | None = None, ok: bool = True, detail: str | None = None) -> dict[str, Any]:
        state = "success" if ok else "error"
        title = "Request finished" if ok else "Request failed"
        event = self.record(trace_id, "python", state, title, detail=detail, source="execution_trace")
        state_doc = self.current_state()
        state_doc["finished"] = True
        state_doc["ok"] = ok
        self._write_state(state_doc)
        return event

    def current_state(self) -> dict[str, Any]:
        if not STATE_FILE.exists():
            return {
                "trace_id": None,
                "components": {name: {"state": "idle", "label": self._label(name), "last_event": None} for name in COMPONENTS},
                "current_step": "Idle",
                "finished": True,
            }
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {
                "trace_id": None,
                "components": {name: {"state": "warning", "label": self._label(name), "last_event": "State unreadable"} for name in COMPONENTS},
                "current_step": "Trace state unreadable",
                "finished": True,
            }

    def events(self, trace_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 10000))
        if not TRACE_FILE.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in TRACE_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if trace_id and item.get("trace_id") != trace_id:
                continue
            rows.append(item)
        return rows[-limit:]

    def reset(self) -> dict[str, Any]:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        return self.current_state()

    def from_result(self, trace_id: str | None, result: dict[str, Any]) -> dict[str, Any]:
        """Create high-level trace events from the coordinator result object."""
        decision = result.get("decision") or {}
        execution = result.get("execution") or {}
        route = result.get("route") or decision.get("route")
        provider = decision.get("provider_name") or result.get("provider_name")
        model = decision.get("model") or result.get("model")
        if route:
            self.record(trace_id, "python", "success", f"Route: {route}", detail=decision.get("reason"), route=route, source="coordinator")
        if provider or model:
            component = "cloud_llm" if provider and "cloud" in str(provider).lower() else "local_llm"
            self.record(trace_id, component, "success", "LLM route used", provider_name=provider, model=model, route=route, source="coordinator")
        raw = json.dumps(result, ensure_ascii=False, default=str).lower()
        if "routing_trace" in raw or "pandora_routing_trace" in raw:
            if "fallback_used\": true" in raw or "fallback_used': true" in raw:
                self.record(trace_id, "local_llm", "warning", "LLM fallback used", source="routing_trace")
            else:
                self.record(trace_id, "local_llm", "success", "LLM analyzer trace present", source="routing_trace")
        if "tool_development" in execution or result.get("proposal_id"):
            self.record(trace_id, "evolution", "success", "Evolution path used", detail="tool_development/proposal", source="coordinator")
        proposal_id = result.get("proposal_id") or execution.get("proposal_id") or ((execution.get("tool_development") or {}).get("proposal") or {}).get("id")
        if proposal_id:
            self.record(trace_id, "proposal", "success", "Proposal created", detail=str(proposal_id), source="coordinator")
        steps = execution.get("steps") if isinstance(execution, dict) else None
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("action_type") == "tool" or step.get("tool_id"):
                    self.record(
                        trace_id,
                        "tool",
                        "success" if step.get("success") else "error",
                        f"Tool: {step.get('tool_id') or 'unknown'}",
                        detail=step.get("error"),
                        duration_ms=step.get("execution_time"),
                        source="worker",
                        payload=step,
                    )
        if result.get("success"):
            self.finish(trace_id, ok=True)
        else:
            self.finish(trace_id, ok=False, detail=result.get("error") or execution.get("error"))
        return self.current_state()

    def _write_state(self, state: dict[str, Any]) -> None:
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _label(self, component: str) -> str:
        return {
            "local_llm": "Local LLM",
            "cloud_llm": "Cloud LLM",
            "python": "Python",
            "tool": "Tool",
            "knowledge": "Knowledge",
            "memory": "Memory",
            "evolution": "Evolution",
            "proposal": "Proposal",
        }.get(component, component)
