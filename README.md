# Pandora Agent – MVP 28.6 Self Observation Engine

Dieser Release baut auf MVP 28.5 auf und ergänzt eine Self Observation Engine.

## Neu in MVP 28.6

- `core/observation/` als eigenes Paket
- Event Bus und Event Logger
- SQLite-basierte Observation Storage-Schicht
- Health-, Statistik-, Runtime- und Export-Funktionen
- CLI-Befehle für Observation
- API-Endpunkte unter `/api/observation/*`
- Maintenance-Link und neue Seite `/observation`
- Genome-Konfiguration für Observation

## Wichtige Grenze

MVP 28.6 sammelt nur Fakten über Pandora selbst.
Es erzeugt keine Verbesserungsvorschläge und verändert keine Core-, Tool-, Skill- oder Genome-Dateien automatisch.
Pattern Recognition beginnt erst mit MVP 28.7.

## CLI Smoke Tests

```bash
python main.py observation-status
python main.py observation-health
python main.py observation-statistics
python main.py observation-record --json '{"component":"tool","event_type":"tool_run","success":true}'
python main.py observation-events --limit 10
```

## API

- `GET /api/observation/status`
- `GET /api/observation/health`
- `GET /api/observation/events`
- `POST /api/observation/events`
- `GET /api/observation/statistics`
- `GET /api/observation/runtime`
- `GET /api/observation/export`

## Release-Hinweis

Runtime-, Test- und Build-Artefakte sind nicht Teil der Clean-ZIP.
