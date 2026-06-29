# MVP 25.5 – Python Orchestrator

Der Python Orchestrator ist die Kontrollschicht nach Request Interpreter und Capability Analyzer.

Er führt nichts aus. Er validiert Empfehlungen und erzeugt einen prüfbaren Plan.

## Aufgaben

- Profil und Route prüfen: `local`, `company`, `cloud`
- empfohlene Quellenräume validieren
- empfohlene Tools und Skills gegen Policies prüfen
- Capability Gaps in reviewbare nächste Schritte übersetzen
- Freigabepflicht markieren
- blockierte Aktionen transparent machen

## Nicht-Aufgaben

Der Orchestrator darf nicht:

- Tools ausführen
- Dateien lesen
- Python-Code erzeugen
- Tools aktivieren
- Core-Dateien ändern
- Releases bauen

## Grundregel

```text
LLM empfiehlt.
Python validiert.
User gibt frei.
Pandora handelt erst danach.
```

## Pipeline

```text
User Request
  ↓
Request Interpreter
  ↓
Capability Analyzer
  ↓
Python Orchestrator
  ↓
Context Builder / Review Workflow / Tool Factory Proposal
```

## CLI

```bash
python main.py python-orchestrator-status
python main.py python-orchestrate "Was war meine letzte Notiz?"
```

## API

```text
GET /api/cognitive/python-orchestrator/status
GET /api/cognitive/python-orchestrator/preview?query=...
```
