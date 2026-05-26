# Pandora Agent MVP 8.2

MVP 8 macht Pandora betreibbarer: Health Monitoring, Watchdog, Benchmarking, Startup Guard und Deployment Manager.

Der aktive Quellcode wird weiterhin nicht automatisch überschrieben. Deployments sind logische Aktivierungen über das Core-Version-System aus MVP 7.

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
python main.py health
python main.py startup-check
```

## Watchdog

Einmalige Prüfung:

```powershell
python main.py watchdog-once
```

Mit automatischem Rollback bei kritischem Zustand:

```powershell
python main.py watchdog-once --auto-rollback
```

Logs:

```powershell
python main.py watchdog-log
python main.py health-log
```

## Benchmark

```powershell
python main.py benchmark
python main.py benchmark-list
```

Gemessen werden aktuell:

- Heartbeat
- Echo Tool
- Calculator Tool
- Echo-Upper Skill

## Deployment

Snapshot erzeugen:

```powershell
python main.py core-snapshot --version-id core_v0_8_0
```

Version deployen:

```powershell
python main.py deploy-version core_v0_8_0
```

Deployen und bei guter Gesundheit als stable markieren:

```powershell
python main.py deploy-version core_v0_8_0 --promote-if-healthy
```

Deployment-Log:

```powershell
python main.py deployment-log
```

## API starten

```powershell
python main.py api
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Neue Endpunkte:

```text
GET  /health
GET  /health/log
POST /watchdog/check
GET  /watchdog/log
POST /benchmark
GET  /benchmark
POST /deployment/{version_id}
GET  /deployment/log
GET  /startup-check
```

## Tests

```powershell
pytest
```

## Architekturstatus

MVP 8 ergänzt Betriebsfähigkeit:

```text
Heartbeat → Health Monitor → Watchdog → Rollback
Snapshot → Validation → Deployment → Health Check → optional Stable Promotion
Startup → Startup Guard → Recovery falls nötig
```

## Wichtig

Noch kein echtes atomisches File-Switching im aktiven Quellcode. Das ist Absicht.

Der nächste sinnvolle Schritt wäre MVP 9:

- kontrollierte Patch-Proposals
- Diff-Review
- Regression-Test-Pipeline
- Staging Deployment
- atomischer Switch
- automatische Rollback-Beobachtungsphase
