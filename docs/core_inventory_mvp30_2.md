# MVP 30.2 – Core Inventory & Cleanup

Ziel: den `core`-Ordner wieder beherrschbar machen, ohne laufende Funktionen blind zu löschen.

## NO KEYWORD ROUTING

Für Routing, Tool-Auswahl, Tool-Gap-Erkennung, Knowledge-/Memory-Auswahl und Tool-Erstellung gilt:

- keine Entscheidung anhand einzelner Wörter in der User-Anfrage
- keine `if "..." in request.lower()`-Routen
- keine Keyword-Tabellen als Hauptentscheidung
- LLM-/Capability-Entscheidung zuerst, Python validiert nur

Keyword-Suche ist nur noch innerhalb bereits freigegebener Suchfunktionen erlaubt, z. B. Vault-Suche, GUI-Filter, Logsuche oder Security-Scans.

## ACTIVE MAIN PATH

Diese Dateien bilden den aktuell bevorzugten Chat-/Agentenfluss:

- `core/coordinator_agent.py`
- `core/chat_service.py`
- `core/capability_orchestrator.py`
- `core/capability_snapshot.py`
- `core/capability_model.py`
- `core/conversation_memory.py`
- `core/cognitive_context_builder.py`
- `core/knowledge_context.py`
- `core/llm_runtime.py`
- `core/llm_chat_responder.py`
- `core/model_router.py`
- `core/models.py`
- `core/tool_registry.py`
- `core/skill_registry.py`
- `core/tool_development_agent.py`
- `core/planner_worker_orchestrator.py`
- `core/planner_agent.py`
- `core/worker_agent.py`

## STRUCTURED-ONLY COMPATIBILITY

Diese Dateien werden noch von älteren CLI/API-/Nebenpfaden importiert, dürfen aber nicht mehr aus User-Text raten:

- `core/action_planner.py`
- `core/capability_detector.py`
- `core/capability_analyzer.py`
- `core/request_interpreter.py`

Änderung in MVP 30.2:

- `ActionPlanner.plan(...)` nutzt nur noch strukturierte Analyse (`suggested_tools`, `suggested_skills`, `risk_level`).
- `CapabilityDetector.detect(...)` nutzt nur noch strukturierte Analyse (`missing_capabilities`, `needed_capabilities`, `suggested_tools`).
- `CapabilityAnalyzer._collect_gaps(...)` erzeugt keine Gaps mehr aus Request-Keywords.
- `RequestInterpreter` bleibt als LLM-Interpreter/Fallback erhalten; der Fallback wählt keine Tools.

## LEGACY / COMPATIBILITY

Diese Datei ist deaktiviert und nur noch als Kompatibilitäts-Shim vorhanden:

- `core/chat_response_router.py`

Status:

- `should_use_tools(...)` gibt immer `False` zurück.
- `deterministic_existing_tool(...)` gibt immer `None` zurück.
- Ersatz ist `core.capability_orchestrator.CapabilityOrchestrator`.

## AREAS TO REVIEW LATER

Diese Bereiche sind Ideen-/MVP-Schichten und sollten erst nach einem stabilen Thinking Core wieder aktiviert oder überarbeitet werden:

- `core/genome/`
- `core/observation/`
- `core/pattern/`
- `core/prioritization/`
- `core/proposal_queue/`
- `core/proposal_generator/`
- `core/proposal_evolution/`
- `core/adaptive_goals/`
- `core/knowledge_evolution/`
- `core/tool_evolution/`
- `core/core_evolution/`
- `core/decision_learning/`
- `core/evolution_dashboard/`

Regel: Diese Bereiche nicht erweitern, bevor Pandora 1.0 / Thinking Core stabil ist.

## NEXT CLEANUP STEP

Nicht löschen, sondern gezielt entkoppeln:

1. API-/CLI-Endpunkte markieren: active, compatibility, experimental.
2. Import-Liste in `main.py` auf Lazy Imports reduzieren.
3. Experimental-/Evolution-Bereiche erst nach Thinking-Core-Release wieder anfassen.
