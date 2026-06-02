# Architektur

Pandora besteht aus Core, Tools, Skills, Memory, Agent Loop, Capability Expansion, Learning und Web-GUI.


## MVP 19.3 – LLM Reliability Layer

Der LLM Reliability Layer sitzt zwischen `LLMRuntime` und den konsumierenden Agenten. Er normalisiert lokale LLM-Antworten, bevor Planner, Tool-Development oder Chat-Logik sie verwenden.

```text
LLM Provider
  ↓
LLMRuntime
  ↓
LLMReliabilityLayer
  ↓
Planner / Chat / Tool Development
```

Aufgaben:

- JSON-Recovery
- Schema-Recovery
- Confidence-Bewertung
- Reasoning-Extraktion
- Reasoning-Persistenz
- Fallback-fähige Planner-Analyse

Wichtiges Prinzip:

`source=llm` oder `success=True` bedeutet nicht mehr blind: Schema passt. Die Reliability-Metadaten zeigen, ob JSON gültig war, ob Schema-Recovery nötig war und mit welcher Confidence Pandora weiterarbeitet.


## MVP 19.3.1 – Capability Gap Routing

The Coordinator now checks `CapabilityDetector` before falling back to normal chat. If a user request describes a missing capability, Pandora routes to `tool_development` and creates a proposal through `ToolDevelopmentAgent`. This prevents friendly LLM chat answers from hiding real missing capabilities.

Current gap examples:

- `word_count` for word-counting requests
- `weather_lookup` for current weather/live weather requests

## MVP 19.3.2 – LLM Capability Gate

Der Coordinator nutzt vor Chat/Planner eine generische Capability-Entscheidung. Der `ToolDevelopmentAgent` fragt das ausgewählte LLM, ob Pandora direkt antworten kann, ein vorhandenes Tool nutzen soll oder eine neue Tool-Fähigkeit benötigt.

Das Ergebnis ist `CapabilityDecision`:

- `can_answer_directly`
- `needs_tool`
- `existing_tool_sufficient`
- `suggested_existing_tool`
- `tool_needed`
- `capability`
- `reason`
- `confidence`

Keyword-basierte Capability-Erkennung bleibt nur als Fallback bei LLM-Fehlern erhalten. Dadurch sind neue Fälle wie Börsenkurse nicht mehr abhängig von fest eingebauten Begriffen.


## MVP 19.3.3 – Deterministic Fast Path

Pandora now avoids LLM routing for conservative, known local tool calls. The Coordinator first checks whether a registered deterministic tool can safely handle the request. If yes, it routes to Planner/Worker directly. The Planner also skips LLM analysis for the same class of requests.

This keeps the LLM capability gate for ambiguous or missing-tool decisions, while making obvious existing-tool tasks faster and cheaper.

```text
Memory Recall
↓
Deterministic existing tool fast path
↓
LLM Capability Gate
↓
Planner / Worker or Tool Development
↓
Chat
```

## MVP 19.3.4 – Single-Pass Capability Gate

The Coordinator now treats capability classification as a single-pass decision. If `decide()` has already called the Tool Development Agent and detected a gap, the resulting gap dictionary is cached and passed into `analyze()` as `precomputed_gap`.

This prevents duplicate LLM calls for the same user message and makes routing more deterministic:

```text
Coordinator.decide()
  ↓ capability gate once
Coordinator.run()
  ↓ reuse precomputed gap
ToolProposalManager
```

