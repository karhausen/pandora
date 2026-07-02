from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LifecycleStep:
    id: str
    title: str
    purpose: str
    order: int
    requires_human: bool = False
    allows_activation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "purpose": self.purpose,
            "order": self.order,
            "requires_human": self.requires_human,
            "allows_activation": self.allows_activation,
        }


class EvolutionLifecycle:
    """Single controlled lifecycle for every Pandora improvement proposal."""

    steps: tuple[LifecycleStep, ...] = (
        LifecycleStep("draft", "Draft", "Idea captured without commitment.", 10),
        LifecycleStep("analysis", "Analysis", "Facts, context and constraints are collected.", 20),
        LifecycleStep("recommendation", "Recommendation", "A recommended direction is prepared.", 30),
        LifecycleStep("proposal", "Proposal", "A concrete change proposal is written.", 40),
        LifecycleStep("review", "Review", "User or maintainer reviews the proposal.", 50, requires_human=True),
        LifecycleStep("tests", "Tests", "Safe checks validate the proposed change.", 60),
        LifecycleStep("approval", "Approval", "Explicit approval is required before activation.", 70, requires_human=True),
        LifecycleStep("activation", "Activation", "Approved change may be applied by controlled code.", 80, allows_activation=True),
        LifecycleStep("learning", "Learning", "Outcome is recorded for future decisions.", 90),
        LifecycleStep("archived", "Archived", "Proposal is closed and preserved for traceability.", 100),
    )

    @classmethod
    def ids(cls) -> list[str]:
        return [step.id for step in cls.steps]

    @classmethod
    def as_dict(cls) -> dict[str, Any]:
        return {
            "kind": "evolution_lifecycle",
            "version": "28.4",
            "single_lifecycle": True,
            "steps": [step.as_dict() for step in cls.steps],
            "activation_requires": ["tests", "approval", "human_decision"],
        }

    @classmethod
    def validate_status(cls, status: str) -> bool:
        return status in cls.ids()
