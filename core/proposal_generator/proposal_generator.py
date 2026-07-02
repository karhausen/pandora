from __future__ import annotations

import json
import re
from typing import Any

from core.genome import EvolutionService
from core.llm_runtime import LLMRuntime
from core.models import LLMRequest, LLMTaskType
from core.proposal_queue import UnifiedProposalQueueManager

from .proposal_prompt import build_proposal_prompt


class ProposalGenerator:
    """Safe draft generator for controlled Pandora evolution proposals.

    MVP 29.0 deliberately produces review-only drafts. It never activates changes,
    never writes generated code and never modifies the core. The LLM is optional:
    when no provider is configured or the call fails, a deterministic heuristic
    draft is generated so CLI/API tests stay local and stable.
    """

    VERSION = "29.0"
    SUPPORTED_TYPES = {"tool", "skill", "knowledge", "workflow", "core", "gui", "prompt", "memory", "personality", "learning"}

    def __init__(self, evolution: EvolutionService | None = None, queue: UnifiedProposalQueueManager | None = None) -> None:
        self.evolution = evolution or EvolutionService()
        self.queue = queue or UnifiedProposalQueueManager()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "proposal_generator_status",
            "version": self.VERSION,
            "ok": True,
            "mode": "draft_only",
            "uses_llm": "optional",
            "fallback": "deterministic_heuristic",
            "activates_changes": False,
            "writes_files": False,
            "requires_review": True,
            "requires_user_approval": True,
            "supported_types": sorted(self.SUPPORTED_TYPES),
            "queue_status": self.queue.status().get("ok", False),
            "genome_valid": self.evolution.validate_genome().get("ok", False),
        }

    def prompt(self, request: str, proposal_type: str | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "kind": "proposal_generator_prompt",
            "version": self.VERSION,
            "prompt": build_proposal_prompt(request, proposal_type=proposal_type, context=context),
            "activates_changes": False,
        }

    def generate(
        self,
        request: str,
        proposal_type: str | None = None,
        context: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        timeout: float = 8.0,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        context = context or {}
        llm_payload: dict[str, Any] | None = None
        llm_error: str | None = None
        if use_llm:
            try:
                req = LLMRequest(
                    task_type=LLMTaskType.REFLECTION,
                    prompt=build_proposal_prompt(request, proposal_type=proposal_type, context=context),
                    provider_name=provider_name,
                    model=model,
                    expect_json=True,
                    timeout=timeout,
                )
                response = LLMRuntime().complete(req)
                text = response.content if hasattr(response, "content") else str(response)
                llm_payload = self._extract_json(text)
            except Exception as exc:  # noqa: BLE001 - safe fallback is intended here
                llm_error = str(exc)

        draft = self._draft_from_llm_or_heuristic(request, proposal_type, context, llm_payload)
        factory_payload = self._to_factory_payload(draft, request, context, llm_used=bool(llm_payload), llm_error=llm_error)
        proposal_result = self.evolution.factory_create(factory_payload)
        proposal = proposal_result.get("proposal", proposal_result)
        return {
            "kind": "proposal_generator_result",
            "version": self.VERSION,
            "ok": True,
            "mode": "draft_only",
            "request": request,
            "proposal": proposal,
            "draft": draft,
            "factory": proposal_result,
            "llm_used": bool(llm_payload),
            "llm_error": llm_error,
            "activates_changes": False,
            "requires_review": True,
            "requires_user_approval": True,
        }

    def generate_and_enqueue(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        generated = self.generate(*args, **kwargs)
        enqueue = self.queue.enqueue(generated["proposal"])
        return {
            "kind": "proposal_generator_enqueue_result",
            "version": self.VERSION,
            "ok": bool(enqueue.get("ok", True)),
            "generated": generated,
            "enqueue": enqueue,
            "activates_changes": False,
            "requires_review": True,
            "requires_user_approval": True,
        }

    def batch_generate(self, items: list[dict[str, Any]], enqueue: bool = False, **kwargs: Any) -> dict[str, Any]:
        results = []
        for item in items:
            request = str(item.get("request") or item.get("title") or item.get("description") or "").strip()
            if not request:
                results.append({"ok": False, "reason": "empty_request", "item": item})
                continue
            proposal_type = item.get("type") or item.get("proposal_type")
            context = item.get("context") or {}
            if enqueue:
                results.append(self.generate_and_enqueue(request, proposal_type=proposal_type, context=context, **kwargs))
            else:
                results.append(self.generate(request, proposal_type=proposal_type, context=context, **kwargs))
        return {"kind": "proposal_generator_batch", "version": self.VERSION, "count": len(results), "results": results, "activates_changes": False}

    def _draft_from_llm_or_heuristic(self, request: str, proposal_type: str | None, context: dict[str, Any], llm_payload: dict[str, Any] | None) -> dict[str, Any]:
        if llm_payload:
            return {
                "type": self._normalize_type(llm_payload.get("type") or proposal_type or self._infer_type(request)),
                "title": str(llm_payload.get("title") or self._title_from_request(request)),
                "description": str(llm_payload.get("description") or request),
                "rationale": str(llm_payload.get("rationale") or "LLM-generated controlled improvement proposal."),
                "expected_benefit": str(llm_payload.get("expected_benefit") or "Verbesserte Systemqualität."),
                "risk": str(llm_payload.get("risk") or "medium").lower(),
                "effort": str(llm_payload.get("effort") or "medium").lower(),
                "confidence": self._confidence(llm_payload.get("confidence"), default=0.72),
                "review_questions": llm_payload.get("review_questions") or ["Ist der Nutzen klar genug?", "Ist das Risiko akzeptabel?"],
                "acceptance_criteria": llm_payload.get("acceptance_criteria") or ["Proposal ist reviewbar.", "Keine automatische Aktivierung."],
            }
        inferred_type = self._normalize_type(proposal_type or self._infer_type(request))
        risk = "high" if inferred_type in {"core", "personality"} else "medium" if inferred_type in {"prompt", "memory", "learning"} else "low"
        return {
            "type": inferred_type,
            "title": self._title_from_request(request),
            "description": request,
            "rationale": "Aus der Anfrage wurde ein kontrollierter Verbesserungsvorschlag abgeleitet.",
            "expected_benefit": self._benefit_for_type(inferred_type),
            "risk": risk,
            "effort": "medium",
            "confidence": 0.62,
            "review_questions": [
                "Ist das Problem konkret genug beschrieben?",
                "Soll dieser Vorschlag in den Review-Prozess aufgenommen werden?",
                "Welche Tests wären vor einer Aktivierung notwendig?",
            ],
            "acceptance_criteria": [
                "Proposal liegt als EvolutionProposal vor.",
                "Proposal ist in der Queue reviewbar.",
                "Keine Änderung wird ohne Benutzerfreigabe aktiviert.",
            ],
        }

    def _to_factory_payload(self, draft: dict[str, Any], request: str, context: dict[str, Any], llm_used: bool, llm_error: str | None) -> dict[str, Any]:
        risk = str(draft.get("risk") or "medium").lower()
        effort = str(draft.get("effort") or "medium").lower()
        priority = 65
        if risk == "high":
            priority -= 10
        if effort == "low":
            priority += 5
        if draft.get("type") == "core":
            priority = max(priority, 70)
        return {
            "type": draft["type"],
            "title": draft["title"],
            "description": draft["description"],
            "source": "proposal_generator",
            "priority": max(0, min(100, priority)),
            "confidence": draft.get("confidence", 0.62),
            "impact": "high" if draft.get("type") in {"core", "tool", "skill", "learning"} else "medium",
            "risk": risk,
            "payload": {
                "mvp": "29.0",
                "generator": "proposal_generator",
                "original_request": request,
                "context": context,
                "draft": draft,
                "llm_used": llm_used,
                "llm_error": llm_error,
                "effort": effort,
                "review_only": True,
            },
        }

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def _normalize_type(self, value: str) -> str:
        normalized = str(value).strip().lower()
        aliases = {"ui": "gui", "interface": "gui", "wissen": "knowledge", "werkzeug": "tool", "fähigkeit": "skill", "faehigkeit": "skill"}
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in self.SUPPORTED_TYPES else "workflow"

    def _infer_type(self, request: str) -> str:
        text = request.lower()
        mapping = [
            ("tool", ("tool", "werkzeug", "cli", "api", "parser", "script")),
            ("skill", ("skill", "fähigkeit", "faehigkeit")),
            ("knowledge", ("wissen", "knowledge", "obsidian", "dokument")),
            ("core", ("core", "architektur", "runtime", "kernel", "main.py")),
            ("gui", ("gui", "ui", "dashboard", "oberfläche", "maintenance")),
            ("prompt", ("prompt", "personality", "kommunikation")),
            ("memory", ("memory", "gedächtnis", "erinner")),
            ("learning", ("learning", "lernen", "entscheidung")),
            ("workflow", ("workflow", "prozess", "ablauf", "review")),
        ]
        for target, keywords in mapping:
            if any(keyword in text for keyword in keywords):
                return target
        return "workflow"

    def _title_from_request(self, request: str) -> str:
        cleaned = " ".join(request.strip().split())
        if len(cleaned) > 72:
            cleaned = cleaned[:69].rstrip() + "..."
        return cleaned or "Controlled Evolution Proposal"

    def _confidence(self, value: Any, default: float) -> float:
        try:
            number = float(value)
            if number > 1:
                number = number / 100.0
            return max(0.0, min(1.0, number))
        except Exception:
            return default

    def _benefit_for_type(self, proposal_type: str) -> str:
        return {
            "tool": "Verbessert konkrete Werkzeugfähigkeit und Automatisierung.",
            "skill": "Erweitert wiederverwendbare Fähigkeiten.",
            "knowledge": "Verbessert Wissensabdeckung und Kontextqualität.",
            "workflow": "Reduziert manuelle Schritte und macht Abläufe nachvollziehbarer.",
            "core": "Verbessert Stabilität oder Architektur des Kernsystems.",
            "gui": "Vereinfacht Bedienung und Review-Fähigkeit.",
            "prompt": "Verbessert Antwortqualität und Rollen-/Stiltreue.",
            "memory": "Verbessert Erinnerung, Kontext und Wiederverwendung.",
            "personality": "Schärft Kommunikationsverhalten und Systemidentität.",
            "learning": "Verbessert Lernen aus Entscheidungen, Fehlern und Erfolgen.",
        }.get(proposal_type, "Verbessert kontrolliert einen Pandora-Systembereich.")
