from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MaintenanceLink:
    label: str
    href: str
    purpose: str
    group: str
    priority: int
    risk: str = "read_only"
    badge: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "href": self.href,
            "purpose": self.purpose,
            "group": self.group,
            "priority": self.priority,
            "risk": self.risk,
            "badge": self.badge,
        }


@dataclass(frozen=True)
class MaintenanceGroup:
    id: str
    title: str
    description: str
    intent: str
    order: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "intent": self.intent,
            "order": self.order,
        }


class MaintenanceCenterService:
    """MVP 28.3: structured Maintenance Center navigation contract.

    The service does not execute maintenance, approve proposals or change
    configuration. It only describes a predictable information architecture for
    the Maintenance GUI so future pages can depend on one stable source.
    """

    version = "28.9"
    codename = "maintenance_center_restructure_with_unified_proposal_queue"

    def groups(self) -> list[MaintenanceGroup]:
        return [
            MaintenanceGroup("overview", "Überblick", "Schneller Systemzustand, offene Punkte und Betriebsampel.", "first_check", 10),
            MaintenanceGroup("decisions", "Entscheidungen", "Alles, was eine bewusste Freigabe oder Bewertung braucht.", "human_control", 20),
            MaintenanceGroup("knowledge", "Wissen", "Knowledge Base, Obsidian und lokale Wissenspflege.", "knowledge_care", 30),
            MaintenanceGroup("capabilities", "Fähigkeiten", "Tools, Skills, Capabilities und deren Lebenszyklus.", "capability_care", 40),
            MaintenanceGroup("configuration", "Konfiguration", "Profile, Modelle, Routing und kognitive Kommunikationsschichten.", "setup", 50),
            MaintenanceGroup("learning", "Lernen & Review", "Reviews, Scheduler, Learning-Metriken und Verbesserungsrückläufe.", "continuous_improvement", 60),
            MaintenanceGroup("evolution", "Evolution", "Genome, Unified Proposal Model, Self Observation, Pattern Recognition, Priorisierung, Proposal Queue, Lifecycle und Evolutionsregeln.", "controlled_evolution", 70),
        ]

    def links(self) -> list[MaintenanceLink]:
        return [
            MaintenanceLink("Operations Cockpit", "/operations-cockpit", "Zentrale Betriebsübersicht mit Health, Issues und empfohlenen nächsten Schritten.", "overview", 10, badge="Start"),
            MaintenanceLink("Operations Health", "/operations-health", "Technische Health-Checks und Diagnoseübersicht.", "overview", 20),
            MaintenanceLink("Operations Issues", "/operations-issues", "Erkannte Probleme, Ursachen und sichere Folgeschritte.", "overview", 30),
            MaintenanceLink("Decision Inbox", "/decision-inbox", "Offene Entscheidungen und Proposal-Handoffs prüfen.", "decisions", 10, risk="human_approval"),
            MaintenanceLink("Action Inbox", "/action-inbox", "Geplante Aktionen und Follow-up-Queue sichtbar machen.", "decisions", 20, risk="human_approval"),
            MaintenanceLink("Approvals", "/approval", "Tool- und Skill-Freigaben bewusst prüfen.", "decisions", 30, risk="human_approval"),
            MaintenanceLink("Knowledge Base", "/knowledge-base", "Wissensbestand suchen und überblicken.", "knowledge", 10),
            MaintenanceLink("Knowledge Editor", "/knowledge-editor", "Wissen kontrolliert anlegen, verschieben und pflegen.", "knowledge", 20, risk="controlled_write"),
            MaintenanceLink("Obsidian Vault", "/obsidian-vault", "Vault-Status, Suche und Export prüfen.", "knowledge", 30),
            MaintenanceLink("Obsidian Import Review", "/obsidian-import-review", "Import-Kandidaten prüfen, bevor Inhalte übernommen werden.", "knowledge", 40, risk="controlled_write"),
            MaintenanceLink("Capability Explorer", "/capability-explorer", "Fähigkeiten, Lücken und Capability Graph nachvollziehen.", "capabilities", 10),
            MaintenanceLink("Tool Center", "/tools-center", "Tools, Status und Aktivierung kontrollieren.", "capabilities", 20, risk="controlled_activation"),
            MaintenanceLink("Skill Center", "/skills-center", "Skills, Kandidaten und Aktivierungen verwalten.", "capabilities", 30, risk="controlled_activation"),
            MaintenanceLink("LLM Profiles", "/llm-profiles", "Provider, Modelle und Profilstatus prüfen.", "configuration", 10, risk="configuration"),
            MaintenanceLink("Cognitive Dashboard", "/cognitive-dashboard", "Identity, Personality, Prompt-Layer und kognitive Pipeline prüfen.", "configuration", 20),
            MaintenanceLink("Night Review", "/night-review", "Review-Pakete und Empfehlungen ansehen.", "learning", 10),
            MaintenanceLink("Review Scheduler", "/review-scheduler", "Geplante Review-Läufe prüfen und manuell anstoßen.", "learning", 20, risk="controlled_run"),
            MaintenanceLink("Workflow Dashboard", "/workflow-dashboard", "Review-to-Action-Workflows und Status verfolgen.", "learning", 30),
            MaintenanceLink("Learning", "/learning", "Learning-Metriken, Muster und Erkenntnisse beobachten.", "learning", 40),
            MaintenanceLink("Evolution", "/evolution", "Pandora Genome, Unified Evolution Model, Lifecycle und Regeln prüfen.", "evolution", 10),
            MaintenanceLink("Self Observation", "/observation", "Events, Health, Runtime-Fakten und Statistiken beobachten – ohne automatische Vorschläge.", "evolution", 20),
            MaintenanceLink("Pattern Recognition", "/pattern", "Wiederkehrende Muster aus Observation Events erkennen – ohne automatische Proposals.", "evolution", 30, badge="Neu"),
            MaintenanceLink("Improvement Prioritization", "/prioritization", "Erkannte Muster nach Nutzen, Risiko, Aufwand und Benutzerwert priorisieren – ohne automatische Proposals.", "evolution", 40),
            MaintenanceLink("Unified Proposal Queue", "/proposal-queue", "Alle Evolution-Proposals zentral filtern, priorisieren und für Review/Freigabe vorbereiten.", "evolution", 50, risk="human_approval", badge="Neu"),
        ]

    def grouped_sections(self) -> list[dict[str, Any]]:
        links = sorted(self.links(), key=lambda item: (item.group, item.priority, item.label))
        by_group: dict[str, list[dict[str, Any]]] = {}
        for link in links:
            by_group.setdefault(link.group, []).append(link.as_dict())
        sections: list[dict[str, Any]] = []
        for group in sorted(self.groups(), key=lambda item: item.order):
            data = group.as_dict()
            data["links"] = by_group.get(group.id, [])
            data["link_count"] = len(data["links"])
            sections.append(data)
        return sections

    def primary_path(self) -> str:
        return "/operations-cockpit"

    def status(self) -> dict[str, Any]:
        sections = self.grouped_sections()
        links = [link.as_dict() for link in self.links()]
        return {
            "kind": "maintenance_center_status",
            "version": self.version,
            "codename": self.codename,
            "mode": "structured_maintenance_center",
            "primary_path": self.primary_path(),
            "group_count": len(sections),
            "link_count": len(links),
            "groups": sections,
            "flat_links": sorted(links, key=lambda item: (item["group"], item["priority"], item["label"])),
            "safety": {
                "read_only_navigation_contract": True,
                "no_maintenance_execution": True,
                "no_proposal_decision": True,
                "no_config_write": True,
                "human_approval_required_for_risk_actions": True,
            },
        }

    def navigation_contract(self) -> dict[str, Any]:
        return {
            "kind": "maintenance_navigation_contract",
            "version": self.version,
            "single_user_entry": "/maintenance",
            "primary_admin_start": self.primary_path(),
            "group_order": [group.id for group in sorted(self.groups(), key=lambda item: item.order)],
            "risk_legend": {
                "read_only": "Nur anzeigen",
                "human_approval": "Entscheidung durch den Nutzer erforderlich",
                "controlled_write": "Schreibend, aber bewusst und begrenzt",
                "controlled_activation": "Aktivierung nur nach Kontrolle",
                "configuration": "Konfiguration bewusst ändern",
                "controlled_run": "Manueller Lauf, kein Hintergrunddienst",
            },
        }
