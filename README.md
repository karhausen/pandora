# Pandora Agent MVP 7.0

MVP 7 ergänzt kontrollierte Core-Evolution:

- Core Version Manager
- Core Snapshots
- Version Manifest
- Active/Stable Version Tracking
- Sandbox Runner
- Smoke Tests
- Activation Manager
- Rollback Manager
- Recovery Manager
- Safe Mode Status
- REST API für Core-Versionen

Der aktive Core wird weiterhin nicht direkt überschrieben. MVP 7 verwaltet Versionen und Aktivierungsentscheidungen, ersetzt aber nicht automatisch Dateien im laufenden Projekt.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Basisprüfung

```powershell
python main.py status
python main.py heartbeat
python main.py tools
python main.py skills
```

## Core Snapshot erzeugen

```powershell
python main.py core-snapshot --version-id core_v0_7_0
```

## Versionen anzeigen

```powershell
python main.py core-versions
python main.py core-active
```

## Version validieren

```powershell
python main.py core-validate core_v0_7_0
```

Dabei werden isoliert ausgeführt:

```text
python main.py status
python main.py heartbeat
python main.py tools
python main.py skills
python main.py run-tool echo --input sandbox
```

Ergebnisse landen in:

```text
core_versions/versions/<VERSION_ID>/heartbeat_results.json
core_versions/versions/<VERSION_ID>/smoke_tests.json
```

## Version aktivieren

```powershell
python main.py core-activate core_v0_7_0 --mark-stable
```

Das aktualisiert:

```text
core_versions/active_version.txt
core_versions/stable_version.txt
```

## Rollback

```powershell
python main.py rollback --reason "Heartbeat failed"
```

## Recovery / Safe Mode

```powershell
python main.py recovery
python main.py recover --reason "manual recovery"
python main.py safe-mode
```

## API starten

```powershell
python main.py api
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Neue API-Endpunkte

```text
POST /core-versions/snapshot
GET  /core-versions
GET  /core-versions/active
POST /core-versions/{version_id}/validate
POST /core-versions/{version_id}/activate
POST /rollback
GET  /recovery/status
POST /recovery/recover
```

## Tests

```powershell
pytest
```

## Wichtige Architekturregel

MVP 7 aktiviert Versionen logisch über Manifest-Dateien. Es überschreibt nicht automatisch den aktiven Quellcode.

Das ist Absicht.

Der nächste Schritt wäre ein kontrollierter Deployment-Schritt:

```text
validated snapshot
→ staged deployment
→ atomic switch
→ heartbeat watch
→ automatic rollback
```

## Protected Core

Diese Dateien gelten als besonders geschützt:

```text
heartbeat.py
rollback_manager.py
recovery.py
security.py
activation_manager.py
version_manager.py
config.py
```

Änderungen daran brauchen explizite Freigabe.
