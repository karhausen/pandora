from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UserGuiNavigationItem:
    label: str
    href: str
    purpose: str
    audience: str = "maintenance"

    def as_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "href": self.href,
            "purpose": self.purpose,
            "audience": self.audience,
        }


class UserGuiSimplificationService:
    """MVP 28.2: keeps the user-facing GUI focused on chat.

    This service is deliberately read-only. It describes the simplified navigation
    contract so the API, docs and regression tests can verify the UX boundary:
    the user page exposes only chat plus one maintenance entry point.
    """

    version = "28.2"
    codename = "user_gui_simplification"

    def user_entry_points(self) -> list[UserGuiNavigationItem]:
        return [
            UserGuiNavigationItem(
                label="Chat",
                href="/",
                purpose="Primary user interaction surface for everyday Pandora requests.",
                audience="user",
            ),
            UserGuiNavigationItem(
                label="Maintenance",
                href="/maintenance",
                purpose="Single controlled entry point for operations, approvals, knowledge, tools, profiles and diagnostics.",
                audience="maintenance",
            ),
        ]

    def maintenance_sections(self) -> list[UserGuiNavigationItem]:
        return [
            UserGuiNavigationItem("Operations Cockpit", "/operations-cockpit", "Health, issues, scheduler and night review overview."),
            UserGuiNavigationItem("Decision Inbox", "/decision-inbox", "Open decisions and proposal handoffs."),
            UserGuiNavigationItem("Action Inbox", "/action-inbox", "Action workflow list and follow-up queue."),
            UserGuiNavigationItem("Knowledge", "/knowledge-base", "Knowledge Base, editor and governance entry."),
            UserGuiNavigationItem("Obsidian", "/obsidian-vault", "Vault status, search and import review."),
            UserGuiNavigationItem("Capabilities", "/capability-explorer", "Capabilities, tools, skills and approval views."),
            UserGuiNavigationItem("LLM Profiles", "/llm-profiles", "Provider, model and routing configuration."),
            UserGuiNavigationItem("Cognitive Dashboard", "/cognitive-dashboard", "Cognitive status, identity and prompt/personality layer."),
            UserGuiNavigationItem("Learning", "/learning", "Observe-only learning metrics and insights."),
        ]

    def status(self) -> dict[str, Any]:
        user_entry_points = [item.as_dict() for item in self.user_entry_points()]
        maintenance_sections = [item.as_dict() for item in self.maintenance_sections()]
        return {
            "kind": "user_gui_simplification_status",
            "version": self.version,
            "codename": self.codename,
            "mode": "chat_first",
            "user_gui_rule": "The user page exposes the chat and exactly one maintenance entry point.",
            "user_entry_point_count": len(user_entry_points),
            "maintenance_entry_point_count": 1,
            "user_entry_points": user_entry_points,
            "maintenance_sections": maintenance_sections,
            "safety": {
                "read_only": True,
                "no_tool_execution": True,
                "no_approval_side_effects": True,
                "no_config_write": True,
            },
        }
