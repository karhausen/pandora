from __future__ import annotations

from .action_planner import ActionPlanner
from .strategy_memory import StrategyMemory


class AdaptivePlanner:
    def __init__(self):
        self.base = ActionPlanner()
        self.strategies = StrategyMemory()

    def plan(self, task: str, analysis: dict):
        # MVP 13 keeps this conservative:
        # Strategy memory may guide future choices, but it does not override safety.
        return self.base.plan(task, analysis)

    def current_strategy(self) -> dict:
        return self.strategies.list()
