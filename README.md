# Local Autonomous Agent MVP1

Ein kleiner, stabiler Python-Core als Fundament für einen später evolvierenden lokalen Agenten.

## Architekturübersicht

Der Core ist absichtlich klein gehalten:

- `AgentCore`: zentrale Verdrahtung der Komponenten
- `Planner`: analysiert Aufgaben und erzeugt einfache Pläne
- `LLMClient`: Provider-Fassade, im MVP sicherer Stub
- `ToolRegistry`: verwaltet Tool-Metadaten
- `ToolExecutor`: lädt und startet registrierte Tools kontrolliert
- `MemoryStore`: Short-Term JSON + SQLite-Dateien für spätere Langzeit-Speicher
- `Heartbeat`: prüft Planner, Memory, Registry, Executor, LLM und Ressourcen
- `RecoveryManager`: empfiehlt Safe Mode bei fehlerhaftem Heartbeat
- `SecurityPolicy`: schützt kritische Core-Dateien
- `FastAPI`: erste Status-, Task-, Tool- und Memory-Endpunkte
- `CLI`: lokale Bedienung

## Projektstruktur

```text
agent/
├── main.py
├── core/
│   ├── agent_core.py
│   ├── api.py
│   ├── config.py
│   ├── heartbeat.py
│   ├── improvement_manager.py
│   ├── llm_client.py
│   ├── memory.py
│   ├── planner.py
│   ├── recovery.py
│   ├── reflection.py
│   ├── rollback.py
│   ├── security.py
│   ├── tool_executor.py
│   └── tool_registry.py
├── tools/
│   ├── calculator.py
│   └── register_builtin_tools.py
├── skills/
├── memory/
├── core_versions/
├── proposals/core_improvements/
├── logs/
├── tests/
├── requirements.txt
└── README.md
```

## MVP-Roadmap

### MVP 1: Überlebensfähiger Core
Enthalten: CLI, Planner, LLM-Stub, Registry, Executor, Memory, Heartbeat, Recovery/Safe-Mode-Entscheidung, Tests.

### MVP 2: Tools
Tool-Metadaten, Beispiel-Tool, Tool-Test, Tool-Bewertung.

### MVP 3: Tool-Erzeugung
Capability-Gap erkennen, Code generieren, statisch prüfen, testen, registrieren.

### MVP 4: Skills
Workflows und Tool-Kombinationen als Skills definieren und ausführen.

### MVP 5: Reflection und Evolution
Task-Auswertung, Vorschläge, Tool-/Skill-Optimierung, Core-Improvement-Proposals.

### MVP 6: API
REST-Endpunkte vollständig ausbauen.

### MVP 7: Core-Versionierung und Rollback
Isolierte Core-Versionen, Smoke-Tests, Heartbeat-Überwachung, automatischer Rollback.

## Sicherheitskonzept

Der aktive Core wird nicht automatisch überschrieben. Besonders geschützt sind:

- `heartbeat.py`
- `rollback.py`
- `recovery.py`
- `security.py`
- `config.py`

Änderungen daran brauchen explizite User-Freigabe. Neue Tools laufen nur nach Registrierung, Validierung und mit Timeout-Konzept. Shell-, Netzwerk- und externe Dateioperationen sind im MVP nicht autonom aktiv.

## Tool-Protokoll

Jedes Tool besitzt Metadaten:

- ID, Name, Beschreibung
- Eingabe- und Ausgabeformat
- Sicherheitsstufe
- Version
- Abhängigkeiten
- Erfolgsquote
- Teststatus
- letzte Nutzung
- Fehlerhistorie
- Modulpfad

Tool-Rückgaben sind standardisiert über `ToolResult(ok, output, error, runtime_ms)`.

## Skill-Protokoll

Noch nicht aktiv in MVP1. Skills werden später unter `/skills` gespeichert und kombinieren Tools, Strategien und Workflows.

## Memory-Konzept

- Short-Term: `memory/short_term.json`
- Long-Term: `memory/long_term.sqlite`
- Episodic: `memory/episodic.sqlite`
- Semantic: `memory/semantic.sqlite`

Im MVP werden Task-Kontext und Reflections gespeichert.

## Heartbeat-Konzept

Der Heartbeat prüft:

1. Planner
2. Memory
3. Tool Registry
4. Tool Executor
5. LLM Client
6. Event Loop Platzhalter
7. Ressourcen-Platzhalter
8. Antwortzeit

Schlägt der Heartbeat fehl, empfiehlt der Core Safe Mode.

## Rollback-Konzept

MVP1 enthält nur den sicheren Platzhalter. Vollständiges Rollback kommt in MVP7:

1. defekte Version deaktivieren
2. letzte stabile Version laden
3. Fehler protokollieren
4. Safe Mode, falls keine stabile Version existiert

## Evolutionsstrategie

Der Core darf Schwächen erkennen und Improvement-Proposals speichern. Er darf den aktiven Core nicht direkt verändern. Neue Core-Versionen werden später isoliert getestet und erst nach Freigabe aktiviert.

## Teststrategie

Aktuell enthalten:

- Heartbeat-Test
- Capability-Gap-Test
- registriertes Beispiel-Tool ausführen

## Starten

```bash
cd agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Tests

```bash
pytest
```

### CLI Status

```bash
python main.py status
```

### Beispiel-Tool registrieren

```bash
python tools/register_builtin_tools.py
python main.py tools
python main.py task "berechne 2+3*4"
```

### API starten

```bash
uvicorn core.api:app --reload
```

Dann:

- `GET /status`
- `GET /heartbeat`
- `POST /task` mit JSON `{"task": "berechne 2+3"}`
- `GET /tools`
- `GET /memory/short-term`

## Wichtige Designentscheidung

Dieses Projekt ist bewusst klein. Der Core ist das Betriebssystem. Tool-Erzeugung, Skill-System, autonome Evolution und Rollback werden später iterativ ergänzt, ohne den aktiven Core unkontrolliert zu verändern.
