# Pandora Agent

Lokaler modularer KI-Agent mit stabilem Core, Tool-/Skill-Evolution, Learning Layer und Web-GUI.

## Projektziel

Pandora soll Aufgaben analysieren, Tools und Skills kontrolliert nutzen, aus Erfahrungen lernen und neue Fähigkeiten sicher vorschlagen.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py status
python main.py api
```

Web-GUI:

```text
http://127.0.0.1:8000
```

## CLI

Wichtige Befehle:

```powershell
python main.py status
python main.py heartbeat
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
python main.py tools
python main.py skills
python main.py learn-from-journal
python main.py recommendations
python main.py docs-generate
python main.py governance-check
```

## API

FastAPI stellt Status-, Agent-, Tool-, Skill-, Capability-, Learning- und Dokumentations-Endpunkte bereit.

## Sicherheit

Der aktive Core darf nicht unkontrolliert überschrieben werden. Kritische Core-Dateien sind geschützt. Neue Tools und Skills entstehen zuerst als Proposal und werden erst nach Validierung und expliziter Aktivierung übernommen.

## Architektur

Siehe `docs/architecture.md`.

## Dokumentation

Weitere Dokumentation befindet sich unter `docs/`.

## Roadmap

Siehe `docs/roadmap.md`.


## MVP 14.1 – Web GUI Fix

- Dashboard vollständig wiederhergestellt
- Agent Run, Heartbeat, Tools, Skills, Journal, Proposals, Learning und Governance sichtbar
- JavaScript nutzt korrekt `provider_name` für `/agent/run`


## MVP 15.0 – Sandbox & Isolation System

Neu:

- ExecutionPolicyManager
- PermissionManager
- ProcessGuard
- ResourceMonitor
- IsolationRunner
- Sandbox
- SandboxLog
- ToolExecutor nutzt standardmäßig Sandbox-Ausführung
- CLI/API für Sandbox-Policies, Sandbox-Logs und isolierte Tool-Ausführung

Beispiele:

```powershell
python main.py sandbox-run-tool calculator --json "{\"expression\":\"2+3*4\"}"
python main.py sandbox-policies
python main.py sandbox-logs
```

Hinweis: MVP 15 bietet Prozess-Isolation und Timeouts. Harte OS-Level CPU/RAM-Limits sind für eine spätere Stufe vorgesehen.


## MVP 16.0 – Real Autonomous Tool Generation

Neu:

- LLMToolGenerator
- ToolCodePrompt
- ToolGenerationRunner
- ToolRepairManager
- ToolGenerationLog
- ToolProposalManager.generate_with_llm()
- CLI/API für LLM-gestützte Tool-Erzeugung

Beispiele:

```powershell
python main.py tool-generate word_count --provider mock
python main.py tool-generation-logs
python main.py tool-proposal-list
python main.py tool-proposal-activate <ID>
```

Sicherheitsregel: Auch MVP 16 aktiviert generierte Tools nicht automatisch. Es erzeugt validierte Proposals.


## MVP 16.1 – Tool Generation Stabilisierung

Neu:

- `tool-generate --no-tests` für schnelle lokale Smoke-Checks
- API-Parameter `run_tests`
- Web-GUI-Panel für Tool Generation
- Tool-Generation-Logs direkt sichtbar
- README/Dashboard nachgezogen

Beispiele:

```powershell
python main.py tool-generate text_reverse --provider mock --no-tests
python main.py tool-generation-logs
```

Für vollständige Validierung ohne `--no-tests`:

```powershell
python main.py tool-generate text_reverse --provider mock
```


## MVP 17.0 – Core Governance & Survival Layer

Neu:

- CoreVersionManager
- CoreSnapshot
- CoreSmokeRunner
- ActivationManager
- RollbackManager
- StabilityMonitor
- Core-Versionen unter `core_versions/`
- CLI/API für Snapshots, Smoke-Tests, Aktivierung und Rollback
- Dashboard-Kachel `Core Status`

Beispiele:

```powershell
python main.py core-status
python main.py core-smoke
python main.py core-snapshot --notes "stable after MVP 17"
python main.py core-versions
python main.py core-rollback
```

Hinweis: MVP 17 markiert Rollbacks und verwaltet Snapshots. Das automatische physische Ersetzen des aktiven Core bleibt bewusst noch manuell, damit der aktive Core nicht unkontrolliert überschrieben wird.


## MVP 17.1 – Reality Check

Neu:

- RealityCheck
- StabilityReporter
- RealityCheckLog
- Dauerlauf-artige Stabilitätsprüfung
- Snapshot-/Memory-Größenreport
- Empfehlungen nach Diagnose
- CLI/API/Dashboard für Reality Checks

Beispiele:

```powershell
python main.py reality-check --iterations 5 --delay 1
python main.py stability-report
python main.py reality-logs
```

Optional mit pytest pro Iteration:

```powershell
python main.py reality-check --iterations 1 --pytest
```


## MVP 18.0 – Planner Agent

Neu:

- PlannerAgent
- TaskPlan / PlanStep Modelle
- TaskPlanStore
- PlannerAgentLog
- CLI/API/Dashboard für strukturierte Planung

Beispiele:

```powershell
python main.py planner-plan "Bitte rechne 2+3*4" --provider mock
python main.py planner-plans
python main.py planner-logs
```

MVP 18 trennt Planung und Ausführung konzeptionell. Der PlannerAgent erzeugt zunächst nur strukturierte Pläne; die Worker-Ausführung folgt in MVP 18.1.


## MVP 18.1 – Worker Agent

Neu:

- WorkerAgent
- WorkerStepResult / TaskExecutionResult Modelle
- TaskExecutionStore
- WorkerAgentLog
- PlannerWorkerOrchestrator
- CLI/API/Dashboard für Plan-Ausführung

Beispiele:

```powershell
python main.py planner-plan "Bitte rechne 2+3*4" --provider mock
python main.py planner-plans
python main.py worker-execute-plan <PLAN_ID>
python main.py planner-worker-run "Bitte rechne 2+3*4" --provider mock
python main.py worker-executions
```

Hinweis: In der Build-Notebook-Umgebung kann der CLI-Smoke mit Sandbox-Subprozessen hängen. Die Unit-/API-Tests prüfen die Worker-Funktionalität erfolgreich.
