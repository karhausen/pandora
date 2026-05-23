from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    ok: bool
    message: str
    runtime_ms: int


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    components: list[ComponentHealth]
    total_runtime_ms: int
    safe_mode_recommended: bool


class Heartbeat:
    def __init__(self, planner, memory, registry, executor, llm_client, timeout_ms: int = 1500):
        self.planner = planner
        self.memory = memory
        self.registry = registry
        self.executor = executor
        self.llm_client = llm_client
        self.timeout_ms = timeout_ms

    def check(self) -> HealthStatus:
        start = time.perf_counter()
        checks = [
            ("planner", self.planner.healthcheck),
            ("memory", self.memory.healthcheck),
            ("tool_registry", self.registry.healthcheck),
            ("tool_executor", self.executor.healthcheck),
            ("llm_client", self.llm_client.healthcheck),
            ("event_loop", lambda: True),
            ("resources", self._resource_check),
        ]
        components: list[ComponentHealth] = []
        for name, fn in checks:
            c_start = time.perf_counter()
            try:
                ok = bool(fn())
                msg = "ok" if ok else "failed"
            except Exception as exc:
                ok = False
                msg = str(exc)
            components.append(ComponentHealth(name, ok, msg, int((time.perf_counter() - c_start) * 1000)))
        total_ms = int((time.perf_counter() - start) * 1000)
        ok = all(c.ok for c in components) and total_ms <= self.timeout_ms
        return HealthStatus(ok, components, total_ms, safe_mode_recommended=not ok)

    def _resource_check(self) -> bool:
        # MVP: portable minimal check. Later: psutil CPU/RAM thresholds.
        return os.getpid() > 0

    def as_dict(self) -> dict:
        return asdict(self.check())
