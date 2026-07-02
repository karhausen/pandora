from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ObservationEvent:
    component: str
    event_type: str
    success: bool = True
    severity: str = "info"
    message: str = ""
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    timestamp: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "component": self.component,
            "event_type": self.event_type,
            "success": self.success,
            "severity": self.severity,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationEvent":
        return cls(
            event_id=str(data.get("event_id") or f"evt_{uuid4().hex[:12]}"),
            timestamp=str(data.get("timestamp") or utc_now()),
            component=str(data.get("component") or "unknown"),
            event_type=str(data.get("event_type") or data.get("event") or "unknown"),
            success=bool(data.get("success", True)),
            severity=str(data.get("severity") or "info"),
            message=str(data.get("message") or ""),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )
