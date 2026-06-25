# Pandora Control Core

MVP 21.0 richtet Pandora wieder auf das ursprüngliche Ziel aus: ein kleiner, stabiler Core als Schaltzentrale.

Der Control Core ist nicht die Wachstumsschicht. Er vermittelt zwischen User, LLM-Routing, Planner, Tools, Skills, Memory und Safety Gate.

## Neue Komponenten

- `core/control_core.py` – zentrale Fassade für CLI/API/UI
- `core/core_status.py` – zentrale Versions- und Statusquelle
- `core/safety_gate.py` – Schutzprüfung für kritische Aktionen
- `core/memory_gateway.py` – kleines stabiles Memory-Interface
- `core/nightly_reflection.py` – Nachtlauf: auswerten, vorschlagen, nichts automatisch aktivieren
- `core/heartbeat.py` – erweiterter Heartbeat für Control Plane

## Regeln

- Core bleibt geschützt.
- Tools, Skills, Workflows und Memory dürfen wachsen.
- Kritische Aktionen benötigen Freigabe.
- Nachtreflexion darf Vorschläge erzeugen, aber keine Änderungen aktivieren.
- UI, CLI und API sollen denselben Control-Core-Pfad nutzen.

## CLI

```bash
python main.py control-status
python main.py control-routes
python main.py control-run "Bitte rechne 2+3*4"
python main.py safety-check core_modify --path core/heartbeat.py
python main.py nightly-reflect --limit 200
python main.py heartbeat
```

## Docker

```bash
docker compose up --build
```

Die API ist dann unter Port 8000 erreichbar. Der Docker Healthcheck nutzt den Heartbeat.
