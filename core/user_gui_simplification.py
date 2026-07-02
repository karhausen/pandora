from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .maintenance_center import MaintenanceCenterService


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

    version = "28.3"
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
        sections: list[UserGuiNavigationItem] = []
        for group in MaintenanceCenterService().grouped_sections():
            first_link = group.get("links", [{}])[0] if group.get("links") else {}
            sections.append(
                UserGuiNavigationItem(
                    label=str(group.get("title", "Maintenance")),
                    href=str(first_link.get("href", "/maintenance")),
                    purpose=str(group.get("description", "Structured maintenance area.")),
                )
            )
        return sections

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
            "maintenance_center": MaintenanceCenterService().status(),
            "safety": {
                "read_only": True,
                "no_tool_execution": True,
                "no_approval_side_effects": True,
                "no_config_write": True,
            },
        }
