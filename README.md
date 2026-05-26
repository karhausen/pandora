# Pandora Agent — MVP 2.0

Lokaler, modularer Python-Agent mit stabilem Core, CLI, Heartbeat, Memory und professionalisiertem Tool-System.

## Status

MVP 2.0 erweitert MVP 1.5 um:

- automatische Tool-Discovery aus `tools/*.py`
- persistente Tool-Runtime-Datenbank `memory/tool_runtime.sqlite`
- gehärteten Tool-Executor mit Timeout, Fehlererfassung und Sicherheitslevel-Prüfung
- Tool-Telemetrie: Runs, Erfolge, Fehler, Laufzeit, Input-/Output-Größe
- erweiterten Heartbeat inklusive Runtime-DB-Check
- CLI-Befehle für Tool-Discovery und Tool-Statistiken
- JSON-Datei-Payloads für stabile Tool-Aufrufe unter PowerShell
- Tests für Discovery, Runtime-Stats und Fehlererfassung

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Wichtige CLI-Befehle

```powershell
python main.py status
python main.py heartbeat
python main.py tools
python main.py tool-list
python main.py tool-discover
python main.py tool-stats
python main.py memory
python main.py safe-mode
```

Tool ausführen:

```powershell
python main.py run-tool echo --input "Hallo Agent"
python main.py run-tool calculator --json '{\"expression\":\"2+3*4\"}'
```

Stabiler unter PowerShell: JSON-Datei nutzen.

```powershell
@'{
  "expression": "2+3*4"
}'@ | Set-Content payload.json

python main.py run-tool calculator --json-file payload.json
```

## Tool-Format

Ein Tool liegt als Python-Datei in `tools/` und braucht mindestens:

```python
from typing import Any

METADATA = {
    "name": "example",
    "description": "Kurzbeschreibung",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object"},
    "safety_level": "low",
}


def run(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}
```

Danach:

```powershell
python main.py tool-discover
python main.py tools
```

## Tool-Runtime-Datenbank

Datei:

```text
memory/tool_runtime.sqlite
```

Tabellen:

- `tool_runs`
- `tool_failures`
- `tool_stats`

Diese Daten sind die Grundlage für spätere Reflection, Tool-Bewertung, Skill-Lernen und kontrollierte Evolution.

## Tests

```powershell
python -m pytest
```

Aktueller Stand: `7 passed`.

## Architekturregel

Der aktive Core wird weiterhin nicht autonom überschrieben. MVP 2 verbessert nur das Tool-System und die Telemetrie. Autonome Tool-Erzeugung kommt erst in MVP 3.
