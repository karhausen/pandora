from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    key: str
    title: str
    action_to_do: str
    description: str


DEFAULT_STEPS = [
    WorkflowStep("review_candidate", "Vorschlag prüfen", "Vorschlag prüfen und nächsten Schritt erlauben", "Initiale Action fachlich prüfen."),
    WorkflowStep("execution_plan", "Ausführungsplan prüfen", "Ausführungsplan prüfen", "Pandora bereitet einen sicheren Plan vor, ohne ihn auszuführen."),
    WorkflowStep("confirm_execution", "Ausführung bestätigen", "Ausführung explizit bestätigen", "Erst dieser Schritt darf die spezialisierte Ausführung starten."),
    WorkflowStep("verify_result", "Ergebnis prüfen", "Ergebnis und Audit prüfen", "Der User bestätigt, ob der Workflow sauber abgeschlossen ist."),
]

OBSIDIAN_IMPORT_STEPS = [
    WorkflowStep("review_candidate", "Obsidian Import-Kandidat prüfen", "Import-Kandidat prüfen", "Prüfen, ob die Vault-Notiz in die Pandora Knowledge Base übernommen werden soll."),
    WorkflowStep("import_plan", "Import-Plan prüfen", "Zielpfad, Metadaten und Konflikte prüfen", "Pandora erstellt einen kontrollierten Import-Plan."),
    WorkflowStep("confirm_import", "Import bestätigen", "Import nach user_knowledge bestätigen", "Import wird nur nach expliziter Bestätigung ausgeführt."),
    WorkflowStep("verify_import", "Import-Ergebnis prüfen", "Import-Ergebnis und Governance prüfen", "Abschlusskontrolle mit Audit/Governance."),
]

CAPABILITY_ACTION_STEPS = [
    WorkflowStep("review_action", "Capability Action prüfen", "Capability Action fachlich prüfen", "Prüfen, ob aus der erkannten Lücke ein nächster Schritt entstehen soll."),
    WorkflowStep("prepare_next_step", "Nächsten Schritt vorbereiten", "Tool/Skill/Knowledge-Plan prüfen", "Pandora erstellt einen prüfbaren Folgeplan."),
    WorkflowStep("confirm_next_step", "Umsetzung bestätigen", "Umsetzung explizit bestätigen", "Keine automatische Tool-/Skill-Aktivierung ohne Bestätigung."),
    WorkflowStep("verify_outcome", "Ergebnis prüfen", "Ergebnis prüfen", "Abschluss und Learnings prüfen."),
]


def steps_for_category(category: str) -> list[WorkflowStep]:
    if category == "obsidian_import_candidate":
        return OBSIDIAN_IMPORT_STEPS
    if category in {"capability_action", "capability_gap", "learning_pattern_action", "learning_insight"}:
        return CAPABILITY_ACTION_STEPS
    return DEFAULT_STEPS


def workflow_id_for(action_id: str) -> str:
    digest = hashlib.sha1(action_id.encode("utf-8")).hexdigest()[:10]
    return f"WF-{digest.upper()}"


def find_step_index(data: dict[str, Any], category: str) -> int:
    value = data.get("workflow_step_index")
    if isinstance(value, int) and value >= 0:
        return value
    # Existing non-workflow proposals are treated as step 0.
    return 0
