# Pandora Agent MVP 6.0

MVP 6 macht aus dem lokalen CLI-Agenten einen lokalen Agent-Service.

Neu:

- FastAPI REST API
- Task Runtime System
- persistente Task-DB
- interne Async-Queue
- Task-Status: QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
- Task-Ausführung über CLI oder API
- Tool-, Skill-, Memory-, Proposal- und Heartbeat-Endpunkte

Der Core wird weiterhin nicht autonom überschrieben.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## CLI prüfen

```powershell
python main.py status
python main.py heartbeat
python main.py tools
python main.py skills
```

## API starten

```powershell
python main.py api
```

Dann öffnen:

```text
http://127.0.0.1:8000/docs
```

## Tool direkt per CLI

```powershell
python main.py run-tool echo --input "Hallo Agent"
python main.py run-tool calculator --json '{\"expression\":\"2+3*4\"}'
```

## Skill direkt per CLI

`payload_skill.json`

```json
{
  "text": "Hallo Agent"
}
```

```powershell
python main.py run-skill echo_then_upper --file payload_skill.json
```

## Task Runtime per CLI

Task einreichen:

```powershell
python main.py submit-task tool --target echo --input "Hallo Task"
```

Task-Liste:

```powershell
python main.py tasks
```

Task manuell ausführen:

```powershell
python main.py task-run <TASK_ID>
```

Task lesen:

```powershell
python main.py task-get <TASK_ID>
```

Task abbrechen:

```powershell
python main.py task-cancel <TASK_ID>
```

## REST API Beispiele

Status:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/status
```

Tool ausführen:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/tools/echo/run `
  -ContentType "application/json" `
  -Body '{"payload":{"text":"Hallo API"}}'
```

Task einreichen:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/tasks `
  -ContentType "application/json" `
  -Body '{"kind":"tool","target":"echo","payload":{"text":"Hallo Queue"}}'
```

Tasks anzeigen:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/tasks
```

## Wichtige API-Endpunkte

```text
GET  /status
GET  /heartbeat
GET  /tools
POST /tools/{tool_id}/run
GET  /skills
POST /skills/{skill_id}/run
POST /task/analyze
POST /tasks
GET  /tasks
GET  /tasks/{task_id}
POST /tasks/{task_id}/execute-now
POST /tasks/{task_id}/cancel
GET  /memory/episodes
GET  /memory/reflections
GET  /proposals
POST /proposals/skills/from-patterns
```

## Tests

```powershell
pytest
```

## Architekturregel

MVP 6 ist Runtime-Infrastruktur.

Es gibt weiterhin keine direkte autonome Core-Modifikation.

Geschützt bleiben:

- Heartbeat
- Rollback
- Recovery
- Security
- Config
- aktiver Core

## Nächster Schritt: MVP 7

MVP 7 sollte Core-Versionierung und Rollback vorbereiten:

- Core-Version-Snapshots
- Version Manifest
- Smoke Tests für neue Versionen
- isolierte Testaktivierung
- Heartbeat-Prüfung neuer Version
- automatischer Rollback
- Safe Mode Recovery
