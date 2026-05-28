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
