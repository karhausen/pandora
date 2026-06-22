from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_config import LLMConfig
from .model_router import ModelRouter
from .user_knowledge_base import AREAS, UserKnowledgeBaseService
from .obsidian_vault import ObsidianVaultService, ObsidianSafetyError


LOCAL_PROVIDER_NAMES = {"mock", "local_fast", "ollama"}
LOCALHOST_MARKERS = ("localhost", "127.0.0.1", "0.0.0.0")


@dataclass
class KnowledgeContextService:
    """Policy-safe context builder for Pandora's user knowledge base.

    The service turns user-owned markdown/text/json files into bounded prompt
    context. It is intentionally conservative: for cloud/company targets it
    excludes every file from private_local_only and returns a blocked counter so
    the UI and execution result can show what happened.
    """

    knowledge: UserKnowledgeBaseService | None = None
    llm_config: LLMConfig | None = None
    max_files: int = 5
    max_chars_per_file: int = 1400
    max_total_chars: int = 6000
    include_obsidian: bool = True

    def __post_init__(self) -> None:
        self.knowledge = self.knowledge or UserKnowledgeBaseService()
        self.llm_config = self.llm_config or LLMConfig()

    def target_for_chat_route(self, provider_name: str | None = None, model: str | None = None) -> dict[str, Any]:
        route = ModelRouter(self.llm_config).route("chat", provider_name_override=provider_name, model_override=model)
        return self.target_for_provider(route.provider_name, model=route.model, route=route.model_dump(mode="json"))

    def target_for_provider(self, provider_name: str | None, *, model: str | None = None, route: dict[str, Any] | None = None) -> dict[str, Any]:
        resolved = provider_name or ""
        provider_type = "unknown"
        base_url = ""
        is_cloud = True
        try:
            cfg = self.llm_config.provider_config(resolved)
            resolved = cfg.get("name", resolved)
            provider_type = cfg.get("type", "unknown")
            base_url = str(cfg.get("base_url") or "")
            if resolved in LOCAL_PROVIDER_NAMES or provider_type in {"mock", "ollama"}:
                is_cloud = False
            elif provider_type == "openai_compatible" and any(marker in base_url for marker in LOCALHOST_MARKERS):
                is_cloud = False
        except Exception:
            # Unknown provider: fail closed and treat as cloud target.
            is_cloud = True
        return {
            "provider_name": resolved or provider_name,
            "model": model,
            "provider_type": provider_type,
            "base_url_kind": "local" if base_url and any(m in base_url for m in LOCALHOST_MARKERS) else ("configured" if base_url else "unknown"),
            "target": "cloud" if is_cloud else "local",
            "cloud_context": is_cloud,
            "route": route or {},
        }

    def build_for_chat(self, query: str, *, provider_name: str | None = None, model: str | None = None, limit: int | None = None) -> dict[str, Any]:
        target = self.target_for_chat_route(provider_name=provider_name, model=model)
        return self.build(query=query, target=target["target"], limit=limit or self.max_files, route_target=target)

    def build(self, *, query: str, target: str = "local", limit: int | None = None, route_target: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_target = (target or "local").strip().lower()
        cloud_context = normalized_target in {"cloud", "company", "company_llm", "cloud_llm", "openai"}
        result = self.knowledge.search(query=query, limit=limit or self.max_files, cloud_context=cloud_context)
        full_result = self.knowledge.search(query=query, limit=200, cloud_context=False) if cloud_context else result
        blocked = [item for item in full_result.get("results", []) if not item.get("cloud_allowed")]

        snippets: list[str] = []
        sources: list[dict[str, Any]] = []
        total_chars = 0
        for item in result.get("results", [])[: limit or self.max_files]:
            area = item["area"]
            rel = item["relative_path"]
            text = self._read_source(area, rel)
            excerpt = self._clip(text, self.max_chars_per_file)
            if not excerpt:
                continue
            header = f"Quelle: user_knowledge/{area}/{rel} | Policy: {item.get('policy')} | Cloud erlaubt: {bool(item.get('cloud_allowed'))}"
            block = f"[{header}]\n{excerpt}"
            if total_chars + len(block) > self.max_total_chars:
                break
            snippets.append(block)
            total_chars += len(block)
            sources.append({
                "area": area,
                "relative_path": rel,
                "policy": item.get("policy"),
                "cloud_allowed": bool(item.get("cloud_allowed")),
                "name": item.get("name"),
                "size_bytes": item.get("size_bytes"),
            })

        obsidian_payload = self._build_obsidian_context(query=query, cloud_context=cloud_context, remaining_files=max(0, (limit or self.max_files) - len(sources)), remaining_chars=max(0, self.max_total_chars - total_chars)) if self.include_obsidian else self._empty_obsidian_payload(cloud_context)
        if obsidian_payload.get("context_text"):
            snippets.append(obsidian_payload["context_text"])
            sources.extend(obsidian_payload.get("sources", []))

        context_text = "\n\n---\n\n".join(snippets)
        return {
            "kind": "knowledge_context",
            "query": query,
            "target": normalized_target,
            "cloud_context": cloud_context,
            "route_target": route_target or {},
            "context_text": context_text,
            "context_chars": len(context_text),
            "source_count": len(sources),
            "sources": sources,
            "blocked_local_only_count": len(blocked) if cloud_context else 0,
            "blocked_obsidian_count": obsidian_payload.get("blocked_count", 0),
            "obsidian": {k: v for k, v in obsidian_payload.items() if k != "context_text"},
            "blocked_local_only_sources": [
                {"area": item.get("area"), "relative_path": item.get("relative_path"), "policy": item.get("policy")}
                for item in blocked[:20]
            ],
            "rule": "private_local_only is included only for local targets; Obsidian context is included for cloud/company targets only when OBSIDIAN_CLOUD_ALLOWED=true",
        }


    def _build_obsidian_context(self, *, query: str, cloud_context: bool, remaining_files: int, remaining_chars: int) -> dict[str, Any]:
        if remaining_files <= 0 or remaining_chars <= 0:
            return self._empty_obsidian_payload(cloud_context)
        try:
            vault = ObsidianVaultService()
            status = vault.status()
            if not status.get("ok"):
                return {**self._empty_obsidian_payload(cloud_context), "enabled": bool(status.get("config", {}).get("enabled")), "status_ok": False, "issues": status.get("issues", [])}
            cloud_allowed = bool(status.get("config", {}).get("cloud_allowed"))
            if cloud_context and not cloud_allowed:
                return {
                    **self._empty_obsidian_payload(cloud_context),
                    "enabled": True,
                    "status_ok": True,
                    "cloud_allowed": False,
                    "blocked_count": 1,
                    "blocked_reason": "OBSIDIAN_CLOUD_ALLOWED=false",
                }
            search = vault.search(query=query, limit=remaining_files, include_content=True)
        except (ObsidianSafetyError, Exception) as exc:
            return {**self._empty_obsidian_payload(cloud_context), "error": str(exc)}

        snippets: list[str] = []
        sources: list[dict[str, Any]] = []
        total = 0
        for item in search.get("results", [])[:remaining_files]:
            text = self._clip(str(item.get("content") or item.get("excerpt") or ""), self.max_chars_per_file)
            if not text:
                continue
            rel = item.get("relative_path")
            header = f"Quelle: obsidian/{rel} | Policy: {'cloud_allowed' if not cloud_context or search.get('cloud_allowed') else 'local_only'} | Cloud erlaubt: {bool(search.get('cloud_allowed'))}"
            block = f"[{header}]\n{text}"
            if total + len(block) > remaining_chars:
                break
            snippets.append(block)
            total += len(block)
            sources.append({
                "source_type": "obsidian",
                "relative_path": rel,
                "title": item.get("title"),
                "tags": item.get("tags", []),
                "wikilinks": item.get("wikilinks", []),
                "cloud_allowed": bool(search.get("cloud_allowed")),
                "score": item.get("score"),
            })
        return {
            "enabled": True,
            "status_ok": True,
            "cloud_allowed": bool(search.get("cloud_allowed")),
            "query": query,
            "source_count": len(sources),
            "sources": sources,
            "context_text": "\n\n---\n\n".join(snippets),
            "blocked_count": 0,
        }

    def _empty_obsidian_payload(self, cloud_context: bool) -> dict[str, Any]:
        return {
            "enabled": False,
            "status_ok": False,
            "cloud_context": cloud_context,
            "cloud_allowed": False,
            "source_count": 0,
            "sources": [],
            "context_text": "",
            "blocked_count": 0,
        }

    def _read_source(self, area: str, relative_path: str) -> str:
        try:
            payload = self.knowledge.show_file(area, relative_path, max_lines=220)
        except Exception:
            return ""
        if not payload.get("found"):
            return ""
        preview = payload.get("preview")
        if isinstance(preview, str):
            return preview
        content = payload.get("content")
        return str(content or "")

    def _clip(self, text: str, max_chars: int) -> str:
        value = (text or "").strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "\n...[gekürzt]"
