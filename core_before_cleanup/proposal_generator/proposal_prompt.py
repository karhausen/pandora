from __future__ import annotations

from typing import Any

SYSTEM_GUARDRAILS = """
Du bist Pandora Proposal Generator. Erzeuge ausschließlich kontrollierte Verbesserungsvorschläge.
Wichtig: Keine Aktivierung, kein Dateischreiben, keine Code-Ausführung, keine Umgehung von Review/Test/User-Freigabe.
Antworte als JSON mit: type, title, description, rationale, expected_benefit, risk, effort, confidence, review_questions, acceptance_criteria.
""".strip()


def build_proposal_prompt(request: str, proposal_type: str | None = None, context: dict[str, Any] | None = None) -> str:
    context = context or {}
    return (
        f"{SYSTEM_GUARDRAILS}\n\n"
        f"Anfrage: {request}\n"
        f"Gewünschter Proposal-Typ: {proposal_type or 'automatisch erkennen'}\n"
        f"Kontext: {context}\n\n"
        "Erzeuge einen sicheren Proposal-Entwurf, der in die Unified Proposal Queue gelegt werden kann."
    )
