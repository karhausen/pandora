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


## MVP 19.4 – Model Router

Der Model Router ist die zentrale Policy-Schicht für Modellwahl. Agenten übergeben nur noch einen Zweck wie `chat`, `tool_selection`, `planning`, `tool_generation` oder `core_review`. Der Router entscheidet anhand von `config/llm/llm_config.json`, welcher Provider und welches Modell genutzt werden.

```text
Agent / Runtime
↓
ModelRouter
↓
LLMConfig provider aliases
↓
local_fast oder cloud_expert
↓
LLMRuntime Client
```

Die wichtigste Trennung:

```text
Alltagsgeschäft / schnelle Entscheidungen -> local_fast
Code-Erzeugung / Tool-Erzeugung / Review -> cloud_expert
```

`LLMRouter` bleibt als Runtime-kompatible Fassade bestehen, delegiert aber an `ModelRouter`. Dadurch bleiben bestehende Aufrufe stabil, während die Modellstrategie zentral konfigurierbar wird.

Sicherheitsregel bleibt unverändert: Cloud-Modelle dürfen Code nur als Proposal erzeugen. Lokale Validierung, Sandbox, Tests und manuelle Aktivierung bleiben Pflicht.

## MVP 19.5 – Cloud Expert Provider

MVP 19.5 führt die Cloud-Expert-Schicht operativ ein. Der Model Router bleibt die zentrale Instanz für die Entscheidung lokal vs. Cloud.

```text
Agent / Tool Development
↓
ModelRouter
↓
CloudExpert / OpenAI Provider
↓
Tool Proposal Manager
↓
Validator / Sandbox / Tests
↓
Manual Activation
```

Routen:

- `chat` → `local_fast`
- `planning` → `local_fast`
- `tool_selection` → `local_fast`
- `tool_generation` → `cloud_expert`
- `code_review` → `cloud_expert`
- `core_review` → `cloud_expert`

Cloud Expert Status prüft nur Konfiguration und Environment. Live-Aufrufe sind explizit (`--live`) und werden nicht in Tests ausgeführt.

Wichtig: Für Tool-Code-Generierung ist der generische Mock-Fallback deaktiviert. Wenn der Cloud-Key fehlt, wird transparent auf deterministische lokale Gerüste zurückgefallen, statt eine Mock-LLM-Antwort als Cloud-Code auszugeben.

## MVP 19.5.1 – Secure LLM Profiles

The Model Router resolves abstract purposes such as `tool_generation` to an active profile. The active profile is selected through safe configuration, while provider URLs, API keys and company model names are read from environment variables.

Load order:

```text
llm_config.template.json
↓
llm_config.json legacy safe config
↓
llm_config.local.json private override
↓
.env / process environment
```

Profile routing:

```text
private cloud_expert → openai
company cloud_expert → company_llm
```

Security rule: repository files may contain placeholder env-variable names, but not real API keys or internal company base URLs. Public API responses use redacted config output.


## MVP 19.5.2 – Profile Manager & Connectivity Tests

Pandora trennt ab MVP 19.5.2 drei Dinge klar:

```text
Git-sichere Template-Konfiguration
↓
Lokales Profil-Override
↓
Secrets aus ENV/.env
```

Der aktive Profile-Name wird über `config/llm/llm_config.local.json` gesetzt. Diese Datei ist lokal und wird nicht versioniert.

```json
{
  "active_profile": "company"
}
```

Der `LLMProfileManager` bietet:

- Profilstatus
- Profilumschaltung
- Provider-Konfigurationsprüfung
- optionalen Live-Smoke-Test
- redaktierte Ausgaben ohne Secrets oder interne URLs

Live-Smoke-Tests sind bewusst opt-in, damit Pandora nicht versehentlich aus dem Firmennetz externe Provider anspricht.

## MVP 19.5.3 – Capability Decision Confidence Normalization

The Capability Gate now separates raw model confidence from Pandora's effective routing confidence.
If a model returns a structurally clear decision such as `tool_needed=true` with a concrete missing capability but reports `confidence=0.0`, Pandora preserves the raw value as `model_confidence` and derives a safe effective `confidence` for routing.

This prevents correct local model reasoning from being discarded only because the model misused the confidence field.


## MVP 19.5.4 – Capability Gate Safety Veto

Capability routing remains LLM-first. However, if the selected local model classifies a clear capability request as direct chat, `ToolDevelopmentAgent` runs a transparent fallback check. If that fallback detects a concrete missing capability, the Coordinator routes to `tool_development` and stores the original LLM decision in `gap.decision`.

This protects Pandora from friendly but wrong chat answers such as asking for text instead of creating or proposing a missing `word_count` tool.


## MVP 19.5.5 – Configuration Refactoring

Static configuration has moved from `memory/` to `config/`. This separates runtime state from deployable configuration and prepares Pandora for Docker volume mapping.

```text
config/  -> static config, profiles, registries, policies
memory/  -> runtime state, chats, facts, logs, reasoning
```

`ConfigManager` centralizes config paths. Legacy files under `memory/` remain readable as migration fallback, but new writes use `config/`.

## MVP 19.6 – Real Tool Design Agent

Pandora erzeugt vor Code-Generierung jetzt einen Tool-Vertrag. Der Tool Development Agent bleibt Initiator, der Tool Design Agent liefert den konkreten Entwurf.

```text
Coordinator
↓
Capability Gate
↓
Tool Development Agent
↓
Tool Design Agent
↓
Tool Proposal Manager
↓
Validator / Tests
↓
Manual Activation
```

Der Design-Schritt beschreibt Input-/Output-Schema, Sicherheitslevel, Netzwerk-/Datei-/Shell-Bedarf, Dependencies, Testfälle und Risiken. Dadurch kann die spätere Cloud-Code-Erzeugung auf einem klaren Vertrag aufsetzen.
