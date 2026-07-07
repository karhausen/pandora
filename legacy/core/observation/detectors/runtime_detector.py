from __future__ import annotations

class Runtime_detector:
    """Placeholder detector contract for MVP 28.6 facts-only observation."""

    detector_id = "runtime"

    def status(self) -> dict:
        return {"detector": self.detector_id, "version": "28.6", "enabled": True, "mode": "facts_only"}
