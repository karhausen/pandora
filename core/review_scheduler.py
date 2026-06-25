from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, UTC, time
from pathlib import Path
from typing import Any

from .config import PROPOSALS_DIR
from .night_review_engine import NightReviewEngine

SCHEDULER_DIR = PROPOSALS_DIR / "review_scheduler"
SCHEDULER_STATE = SCHEDULER_DIR / "state.json"


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool
    hour: int
    minute: int
    limit: int
    create_actions: bool

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        enabled = os.getenv("PANDORA_REVIEW_SCHEDULER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        raw_time = os.getenv("PANDORA_NIGHT_REVIEW_TIME", "02:00").strip() or "02:00"
        try:
            hour_s, minute_s = raw_time.split(":", 1)
            hour = max(0, min(23, int(hour_s)))
            minute = max(0, min(59, int(minute_s)))
        except Exception:
            hour, minute = 2, 0
        try:
            limit = max(1, int(os.getenv("PANDORA_NIGHT_REVIEW_LIMIT", "200")))
        except Exception:
            limit = 200
        create_actions = os.getenv("PANDORA_NIGHT_REVIEW_CREATE_ACTIONS", "true").strip().lower() not in {"0", "false", "no", "off"}
        return cls(enabled=enabled, hour=hour, minute=minute, limit=limit, create_actions=create_actions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "time": f"{self.hour:02d}:{self.minute:02d}",
            "limit": self.limit,
            "create_actions": self.create_actions,
        }


class ReviewSchedulerService:
    """Small, explicit scheduler facade for Night Review.

    This service does not start a background daemon. It tells callers whether a
    scheduled run is due and can execute one run when invoked by CLI, GUI, cron,
    Windows Task Scheduler, Docker cron, or a future Pandora service loop.
    """

    version = "mvp-24.9-review-scheduler-manual-run-center"

    def __init__(self, *, state_path: Path = SCHEDULER_STATE, engine: NightReviewEngine | None = None) -> None:
        self.state_path = state_path
        self.engine = engine or NightReviewEngine()
        self.config = SchedulerConfig.from_env()

    def status(self) -> dict[str, Any]:
        state = self._read_state()
        last_run = state.get("last_run")
        due = self.is_due(now=datetime.now(UTC), state=state)
        return {
            "kind": "review_scheduler_status",
            "version": self.version,
            "generated_at": datetime.now(UTC).isoformat(),
            "config": self.config.as_dict(),
            "state_path": str(self.state_path),
            "last_run": last_run,
            "last_result": state.get("last_result"),
            "run_count": len(state.get("runs", [])),
            "due": due,
            "safety": self.safety(),
        }

    def is_due(self, *, now: datetime | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        state = state or self._read_state()
        scheduled = time(hour=self.config.hour, minute=self.config.minute)
        after_scheduled_time = now.time() >= scheduled
        last_run = state.get("last_run")
        ran_today = False
        if isinstance(last_run, str):
            try:
                ran_today = datetime.fromisoformat(last_run).date() == now.date()
            except Exception:
                ran_today = False
        due = self.config.enabled and after_scheduled_time and not ran_today
        reason = "due" if due else "not_due"
        if not self.config.enabled:
            reason = "scheduler_disabled"
        elif not after_scheduled_time:
            reason = "before_scheduled_time"
        elif ran_today:
            reason = "already_ran_today"
        return {
            "due": due,
            "reason": reason,
            "now_utc": now.isoformat(),
            "scheduled_time_utc": f"{self.config.hour:02d}:{self.config.minute:02d}",
            "ran_today": ran_today,
        }

    def run_manual(self, *, limit: int | None = None, write: bool = True, create_actions: bool | None = None) -> dict[str, Any]:
        limit = int(limit or self.config.limit)
        if create_actions is None:
            create_actions = self.config.create_actions
        result = self.engine.run(limit=limit, write=write, create_actions=bool(create_actions))
        self._record_run(trigger="manual", result=result, write=write, limit=limit, create_actions=bool(create_actions))
        return {"kind": "review_scheduler_manual_run", "ok": True, "trigger": "manual", "result": result, "safety": self.safety()}

    def run_if_due(self, *, force: bool = False) -> dict[str, Any]:
        state = self._read_state()
        due = self.is_due(state=state)
        if not force and not due.get("due"):
            return {"kind": "review_scheduler_due_run", "ok": False, "skipped": True, "due": due, "safety": self.safety()}
        result = self.engine.run(limit=self.config.limit, write=True, create_actions=self.config.create_actions)
        self._record_run(trigger="scheduled" if not force else "forced", result=result, write=True, limit=self.config.limit, create_actions=self.config.create_actions)
        return {"kind": "review_scheduler_due_run", "ok": True, "skipped": False, "due": due, "result": result, "safety": self.safety()}

    def history(self, *, limit: int = 50) -> dict[str, Any]:
        state = self._read_state()
        runs = list(reversed(state.get("runs", [])))[:limit]
        return {"kind": "review_scheduler_history", "version": self.version, "count": len(runs), "runs": runs, "state_path": str(self.state_path)}

    def safety(self) -> dict[str, bool]:
        return {
            "background_daemon": False,
            "manual_or_external_trigger": True,
            "auto_execute_actions": False,
            "creates_reviewable_actions_only": True,
            "core_changes": False,
        }

    def _record_run(self, *, trigger: str, result: dict[str, Any], write: bool, limit: int, create_actions: bool) -> None:
        state = self._read_state()
        now = datetime.now(UTC).isoformat()
        report = (result.get("report") if isinstance(result, dict) else None) or {}
        run = {
            "ts": now,
            "trigger": trigger,
            "write": write,
            "limit": limit,
            "create_actions": create_actions,
            "ok": bool(result.get("report")) if isinstance(result, dict) else False,
            "report_id": report.get("id"),
            "recommendation_count": report.get("recommendation_count", 0),
        }
        runs = state.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        runs.append(run)
        state.update({"kind": "review_scheduler_state", "version": self.version, "last_run": now, "last_result": run, "runs": runs[-500:]})
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _read_state(self) -> dict[str, Any]:
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
        return {"kind": "review_scheduler_state", "version": self.version, "runs": []}
