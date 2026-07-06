# MVP 30.8 – Core Runtime Map & Cleanup

Stand: basiert auf MVP 30.7.

## Ziel

Kein neues Feature. Dieser Schritt dokumentiert und stabilisiert den aktuell aktiven Kern:

- LLM-led Route Registry
- Conversation Loop
- Vault/Memory-Kontext
- direkter LLM-Chat

Tools, Skills, Capability Gap, Tool Factory und Evolution bleiben deaktiviert.

## Aktiver Runtime-Pfad

```text
User
  -> ChatService
  -> LLMRoutePlanner / Prompt Builder
  -> LLM entscheidet Route
  -> RouteRegistry dispatcht
  -> Route führt aus: direct_answer | vault_search | memory_search | clarify_user
  -> RouteContextBuilder bündelt Kontext
  -> LLM formuliert Antwort
  -> ChatService speichert Session
  -> API/User
```

## Aktiv im Zielpfad

```text
core.chat_service
core.llm_route_registry
core.route_context_builder
core.llm_chat_responder
core.knowledge_context
core.obsidian_search
core.obsidian_vault
core.user_knowledge_base
core.conversation_memory
core.working_memory
core.llm_runtime
core.model_router
core.llm_config
core.llm_profile_manager
core.cloud_expert
core.api
core.models
core.config_manager
```

## Bewusst deaktiviert / Legacy-Quarantäne per Policy

Diese Module bleiben im Projekt, werden aber für den aktuellen Chat-Hauptpfad nicht verwendet. Sie dürfen nicht wieder direkt in den Hauptpfad eingebaut werden, bevor Vault + LLM stabil RELEASED sind.

```text
core.chat_response_router
core.action_planner
core.capability_detector
core.capability_analyzer
core.capability_orchestrator
core.planner_worker_orchestrator
core.tool_development_agent
core.tool_generator
core.tool_executor
core.skill_executor
core.capability_gap_pipeline
core.capability_gap_intelligence
core.cognitive_reasoning_layer
```

## Statische Inventur

```json
{
  "module_count": 266,
  "active_runtime": [
    "core.chat_service",
    "core.llm_route_registry",
    "core.route_context_builder",
    "core.llm_chat_responder",
    "core.knowledge_context",
    "core.obsidian_search",
    "core.obsidian_vault",
    "core.user_knowledge_base",
    "core.conversation_memory",
    "core.working_memory",
    "core.llm_runtime",
    "core.model_router",
    "core.llm_config",
    "core.llm_profile_manager",
    "core.cloud_expert",
    "core.api",
    "core.models",
    "core.config_manager"
  ],
  "legacy_disabled_or_quarantined_by_policy": [
    "core.chat_response_router",
    "core.action_planner",
    "core.capability_detector",
    "core.capability_analyzer",
    "core.capability_orchestrator",
    "core.planner_worker_orchestrator",
    "core.tool_development_agent",
    "core.tool_generator",
    "core.tool_executor",
    "core.skill_executor",
    "core.capability_gap_pipeline",
    "core.capability_gap_intelligence",
    "core.cognitive_reasoning_layer"
  ],
  "static_unreferenced_count": 130,
  "tests_count": 13
}
```

Hinweis: Eine statisch unreferenzierte Datei ist nicht automatisch löschbar, weil alte CLI/API-Endpunkte, dynamische Imports oder spätere MVPs sie noch referenzieren können. Deshalb wurde in diesem MVP nicht riskant gelöscht.

## Cleanup-Regel ab jetzt

1. Der Router entscheidet nicht inhaltlich.
2. Nur das LLM wählt eine Route.
3. Python validiert und dispatcht.
4. Alte Tool-/Capability-/Evolution-Pfade bleiben aus dem Chat-Hauptpfad draußen.
5. Neue Routen werden ausschließlich über die RouteRegistry ergänzt.
