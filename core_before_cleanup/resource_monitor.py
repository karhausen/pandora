from __future__ import annotations

import os
import time


class ResourceMonitor:
    def snapshot(self) -> dict:
        # Lightweight and cross-platform. Real per-process limits come later.
        return {
            "pid": os.getpid(),
            "time": time.time(),
        }
