from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .knowledge_context import KnowledgeContextService
from .llm_config import LLMConfig
from .model_router import ModelRouter


@dataclass
class CognitiveContextBuilder:
    """Builds policy-safe pre-LLM context for chat/reasoning.

    The LLM should never be expected to read local files itself. This builder
    resolves the active chat route, determines the privacy target
    (local/company/cloud), queries allowed knowledge providers and returns both
    prompt context and diagnostics explaining included or blocked sources.
    """

    knowledge_context: KnowledgeContextService | None = None
    llm_config: LLMConfig | None = None

    def __post_init__(self) -> None:
        self.llm_config = self.llm_config or LLMConfig()
        self.knowledge_context = self.knowledge_context or KnowledgeContextService(llm_config=self.llm_config)

    def build_for_chat(self, query: str, *, provider_name: str | None = None, model: str | None = None, limit: int | None = None) -> dict[str, Any]:
        route = ModelRouter(self.llm_config).route("chat", provider_name_override=provider_name, model_override=model)
        target = self.knowledge_context.target_for_provider(route.provider_name, model=route.model, route=route.model_dump(mode="json"))
        payload = self.knowledge_context.build(query=query, target=target["target"], limit=limit, route_target=target)
        return {
            "kind": "cognitive_context",
            "query": query,
            "purpose": "chat",
            "route_target": target,
            "target": payload.get("target"),
            "context_text": payload.get("context_text", ""),
            "context_chars": payload.get("context_chars", 0),
            "sources": payload.get("sources", []),
            "source_count": payload.get("source_count", 0),
            "context_ranking": payload.get("context_ranking", {}),
            "policy": payload.get("policy", {}),
            "diagnostics": {
                "knowledge_context": payload,
                "blocked_local_only_count": payload.get("blocked_local_only_count", 0),
                "blocked_obsidian_count": payload.get("blocked_obsidian_count", 0),
                "obsidian": payload.get("obsidian", {}),
                "context_ranking": payload.get("context_ranking", {}),
            },
        }

    def status(self) -> dict[str, Any]:
        route = ModelRouter(self.llm_config).route("chat")
        target = self.knowledge_context.target_for_provider(route.provider_name, model=route.model, route=route.model_dump(mode="json"))
        return {
            "kind": "cognitive_context_builder_status",
            "ok": True,
            "chat_route": route.model_dump(mode="json"),
            "target": target,
            "providers": ["user_knowledge", "obsidian_vault", "conversation_memory"],
            "pipeline": ["collect_candidates", "rank", "duplicate_removal", "budget", "prompt_context"],
            "policy_levels": ["local_only", "company_allowed", "cloud_allowed"],
        }
