from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import json

from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType


@dataclass(frozen=True)
class RouteSpec:
    id: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    requires_approval: bool = False
    safety_note: str = ""


@dataclass
class RouteRequest:
    route: str
    input: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RouteRequest":
        route = str(payload.get("route") or payload.get("action") or "clarify_user").strip()
        raw_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
        return cls(
            route=route,
            input=raw_input,
            reason=str(payload.get("reason") or ""),
            confidence=_safe_float(payload.get("confidence"), 0.0),
        )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


class RouteHandler(Protocol):
    def execute(self, task: str, route_request: RouteRequest, service: Any, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        ...


class PromptBuilder:
    """Builds the route-selection prompt.

    This builder gives the LLM the list of currently available routes and the
    rules for requesting sources. It does not decide which route is correct.
    """

    def build(self, task: str, routes: list[RouteSpec], history: list[dict] | None = None) -> str:
        active_routes = [r for r in routes if r.enabled]
        route_payload = [
            {
                "route": r.id,
                "description": r.description,
                "input_schema": r.input_schema,
                "requires_approval": r.requires_approval,
                "safety_note": r.safety_note,
            }
            for r in active_routes
        ]
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in (history or [])[-6:]
        )
        return (
            "Du bist Pandoras LLM-Routenplaner. Du entscheidest fachlich, welche Quelle oder Route fuer die Anfrage benoetigt wird.\n"
            "Der Python-Router entscheidet NICHT fachlich; er validiert und fuehrt nur deine Route aus.\n"
            "Waehle genau EINE Route aus der Liste. Tools, Tool-Entwicklung, Planner/Worker und Capability-Gaps sind in diesem MVP deaktiviert.\n"
            "Wenn die Frage sich auf gespeichertes Wissen, Projektnotizen, Obsidian, vorhandene Prompts, Todos, Roadmaps oder Benutzerdaten bezieht, fordere Kontext ueber eine Knowledge-/Vault-Route an.\n"
            "Wenn allgemeines Weltwissen ohne gespeicherte Benutzerdaten reicht, waehle direct_answer.\n"
            "Antworte NUR mit gueltigem JSON in diesem Schema:\n"
            "{\"route\": string, \"input\": object, \"reason\": string, \"confidence\": number}\n\n"
            f"Verfuegbare Routen:\n{json.dumps(route_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Bisheriger Verlauf:\n{history_text}\n\n"
            f"Nutzeranfrage:\n{task}"
        )


class LLMRoutePlanner:
    """Asks the LLM for a route request. It does not execute anything."""

    def __init__(self, llm: LLMRuntime | None = None, prompt_builder: PromptBuilder | None = None):
        self.llm = llm or LLMRuntime()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def choose_route(
        self,
        task: str,
        routes: list[RouteSpec],
        *,
        history: list[dict] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        prompt = self.prompt_builder.build(task, routes, history=history)
        # Route selection is a real LLM decision. Do not fall back to the mock
        # client for live chat requests: a mock clarification would be another
        # hidden Python-side decision and would hide the actual configuration
        # problem. When no provider is explicitly selected, use the active
        # profile's cloud expert for this small JSON decision.
        response = self.llm.complete(LLMRequest(
            task_type=LLMTaskType.PLANNING,
            prompt=prompt,
            system_prompt="Return only valid JSON for Pandora's route request schema.",
            provider_name=provider_name or "cloud_expert",
            model=model,
            expect_json=True,
            timeout=30.0,
            allow_provider_fallback=False,
            context={"task": task, "purpose": "llm_led_route_selection"},
        ))
        if not response.success:
            return {
                "success": False,
                "route_request": RouteRequest(route="clarify_user", input={"question": "Ich konnte gerade nicht sicher bestimmen, welche Quelle ich nutzen soll."}, reason=response.error or "LLM route selection failed"),
                "raw": response.raw,
                "error": response.error,
            }
        try:
            route_request = RouteRequest.from_payload(response.parsed_json or {})
        except Exception as exc:
            return {
                "success": False,
                "route_request": RouteRequest(route="clarify_user", input={"question": "Die Routenentscheidung war unlesbar."}, reason=str(exc)),
                "raw": response.raw,
                "error": str(exc),
            }
        return {
            "success": True,
            "route_request": route_request,
            "raw": response.raw,
            "provider_name": response.provider_name,
            "model": response.model,
        }


class DirectAnswerRoute:
    def execute(self, task: str, route_request: RouteRequest, service: Any, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        return {"kind": "direct_answer", "context_text": "", "sources": [], "source_count": 0}


class VaultSearchRoute:
    def execute(self, task: str, route_request: RouteRequest, service: Any, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        query = str(route_request.input.get("query") or task)
        payload = service.knowledge_context.build_for_chat(query, provider_name=provider_name, model=model)
        payload["kind"] = "vault_search"
        payload["route_query"] = query
        return payload


class MemorySearchRoute:
    def execute(self, task: str, route_request: RouteRequest, service: Any, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        session_id = route_request.input.get("session_id") or getattr(service, "_active_session_id", None)
        memory_answer = service.memory.answer_from_memory(task)
        context_text = memory_answer or ""
        return {
            "kind": "memory_search",
            "context_text": context_text,
            "sources": [{"source_type": "conversation_memory"}] if context_text else [],
            "source_count": 1 if context_text else 0,
            "session_id": session_id,
        }


class ClarifyUserRoute:
    def execute(self, task: str, route_request: RouteRequest, service: Any, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        question = route_request.input.get("question") or "Ich brauche dazu eine kurze Klärung."
        return {"kind": "clarify_user", "question": str(question), "context_text": "", "sources": [], "source_count": 0}


class RouteRegistry:
    """Expandable registry. The router dispatches by LLM-requested route id only."""

    def __init__(self):
        self._specs: dict[str, RouteSpec] = {}
        self._handlers: dict[str, RouteHandler] = {}
        self.register(RouteSpec(
            id="direct_answer",
            description="Use this for a general question that can be answered without stored user knowledge.",
            input_schema={},
            enabled=True,
            safety_note="No local files or tools are accessed.",
        ), DirectAnswerRoute())
        self.register(RouteSpec(
            id="vault_search",
            description="Search policy-approved user knowledge, Obsidian/Vault notes and project documents, then answer with that context.",
            input_schema={"query": "string"},
            enabled=True,
            safety_note="Only policy-approved context may be sent to the selected LLM target.",
        ), VaultSearchRoute())
        self.register(RouteSpec(
            id="memory_search",
            description="Use conversation memory when the answer depends on remembered conversation facts.",
            input_schema={"query": "string"},
            enabled=True,
            safety_note="Uses stored conversation memory only.",
        ), MemorySearchRoute())
        self.register(RouteSpec(
            id="clarify_user",
            description="Ask a short clarification question when no safe route can be selected.",
            input_schema={"question": "string"},
            enabled=True,
            safety_note="No external action is performed.",
        ), ClarifyUserRoute())

        # Future routes are intentionally disabled for this MVP. They are listed
        # as extension points, but not offered as executable active routes.
        self._specs["tool_execute"] = RouteSpec(
            id="tool_execute",
            description="Future route: execute an existing approved tool.",
            input_schema={"tool_id": "string", "payload": "object"},
            enabled=False,
            safety_note="Disabled in MVP 30.4.",
        )
        self._specs["skill_execute"] = RouteSpec(
            id="skill_execute",
            description="Future route: execute an approved skill.",
            input_schema={"skill_id": "string", "payload": "object"},
            enabled=False,
            safety_note="Disabled in MVP 30.4.",
        )
        self._specs["capability_gap"] = RouteSpec(
            id="capability_gap",
            description="Future route: create a reviewable capability gap proposal.",
            input_schema={"capability": "string", "reason": "string"},
            enabled=False,
            safety_note="Disabled in MVP 30.4.",
        )

    def register(self, spec: RouteSpec, handler: RouteHandler) -> None:
        self._specs[spec.id] = spec
        self._handlers[spec.id] = handler

    def available_specs(self) -> list[RouteSpec]:
        return [spec for spec in self._specs.values() if spec.enabled]

    def all_specs(self) -> list[RouteSpec]:
        return list(self._specs.values())

    def dispatch(self, route_request: RouteRequest, task: str, service: Any, *, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        spec = self._specs.get(route_request.route)
        if not spec:
            route_request = RouteRequest(route="clarify_user", input={"question": "Diese Route ist nicht registriert."}, reason="unknown_route")
            spec = self._specs["clarify_user"]
        if not spec.enabled:
            route_request = RouteRequest(route="clarify_user", input={"question": "Diese Route ist in diesem MVP noch deaktiviert."}, reason="disabled_route")
            spec = self._specs["clarify_user"]
        handler = self._handlers.get(spec.id)
        if not handler:
            route_request = RouteRequest(route="clarify_user", input={"question": "Für diese Route gibt es keinen Handler."}, reason="missing_handler")
            handler = self._handlers["clarify_user"]
            spec = self._specs["clarify_user"]
        result = handler.execute(task, route_request, service, provider_name=provider_name, model=model)
        result["route"] = spec.id
        result["route_request"] = {
            "route": route_request.route,
            "input": route_request.input,
            "reason": route_request.reason,
            "confidence": route_request.confidence,
        }
        return result
