from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollbackResult:
    ok: bool
    message: str


class RollbackManager:
    """MVP placeholder. Full version activation/rollback comes in MVP7."""

    def rollback_to_last_stable(self) -> RollbackResult:
        return RollbackResult(False, "MVP1: no core version store active yet; start safe mode")
