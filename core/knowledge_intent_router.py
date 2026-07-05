from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm_runtime import LLMRuntime
from .models import LLMRequest, LLMTaskType


@dataclass
class KnowledgeIntentDecision:
    needs_knowledge: bool
    confidence: float
    reason: str
    mode: str
    context: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "needs_knowledge": self.needs_knowledge,
            "confidence": self.confidence,
            "reason": self.reason,
            "mode": self.mode,
            "context": self.context or {},
        }


@dataclass
class KnowledgeIntentRouter:
    """Decides only whether approved user knowledge is needed for an answer.

    MVP 30.4 deliberately avoids tool routing, capability gaps and planner
    execution. The only decision is:
      - needs_knowledge=True  -> collect approved Vault/Knowledge context, then ask LLM
      - needs_knowledge=False -> ask LLM directly

    The decision is semantic. There are no request-keyword routing tables here.
    """

    llm_runtime: LLMRuntime | None = None

    def __post_init__(self) -> None:
        self.llm_runtime = self.llm_runtime or LLMRuntime()

    def decide(self, task: str, *, provider_name: str | None = None, model: str | None = None) -> KnowledgeIntentDecision:
        system_prompt = (
            "You are Pandora's knowledge-intent router for MVP 30.4. Return ONLY valid JSON. "
            "Do not answer the user. Do not choose tools. Do not propose new capabilities. "
            "Your only decision is whether Pandora must use its stored user knowledge, memory, or Vault before answering. "
            "Choose needs_knowledge=true when the user asks about their own stored notes, project state, todos, documents, test prompts, previous decisions, roadmap, or anything that depends on Pandora's private/user knowledge stores. "
            "Choose needs_knowledge=false for general world knowledge, explanations, definitions, casual chat, and questions that do not require the user's stored data. "
            "Do not route by a single keyword. Decide by the meaning of the whole request. "
            "Schema: {needs_knowledge:boolean, confidence:number, reason:string}."
        )
        request = LLMRequest(
            task_type=LLMTaskType.PLANNING,
            prompt=task,
            system_prompt=system_prompt,
            context={"task": task, "allowed_paths": ["knowledge_then_llm", "direct_llm"]},
            provider_name=provider_name,
            model=model,
            expect_json=True,
            timeout=8.0,
            allow_provider_fallback=True,
        )
        response = self.llm_runtime.complete(request)
        if response.success and isinstance(response.parsed_json, dict):
            data = response.parsed_json
            return KnowledgeIntentDecision(
                needs_knowledge=bool(data.get("needs_knowledge")),
                confidence=float(data.get("confidence") or 0.5),
                reason=str(data.get("reason") or "Semantic knowledge intent decision."),
                mode="llm_knowledge_intent",
                context={"raw": data, "provider_name": response.provider_name, "model": response.model},
            )
        # Safe fallback: do not claim knowledge. Chat can still answer general
        # questions, while explicit knowledge questions will be caught by tests
        # using real LLM or by manual regression.
        return KnowledgeIntentDecision(
            needs_knowledge=False,
            confidence=0.2,
            reason="Knowledge intent decision unavailable; using direct LLM without loading Vault context.",
            mode="fallback_direct_llm",
            context={"error": response.error},
        )
