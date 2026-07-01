# MVP 27.2 – Adaptive Tool Selection

## Ziel

Pandora soll Tools nicht starr per Keyword auswählen. Der Cognitive Planner darf benötigte Werkzeuge oder Fähigkeiten empfehlen. Python normalisiert diese Empfehlung, prüft die Tool Registry, bewertet vorhandene Tools und erkennt Tool-Gaps.

## Grundregel

```text
LLM empfiehlt Tool-Raum oder fehlende Fähigkeit.
Python prüft Registry, Status, Security-Level und Profil.
Kein Tool wird automatisch ausgeführt.
Kein Code wird automatisch generiert.
```

## Ablauf

```text
User Request
  ↓
Cognitive Plan
  ↓
Adaptive Tool Selector
  ↓
Tool Registry prüfen
  ↓
Ranking vorhandener Tools
  ↓
Tool Gap erkennen
  ↓
Central Decision / Approval Workflow
```

## CLI

```bash
python main.py adaptive-tool-selection-status
python main.py adaptive-tool-select "Bitte rechne 2+3*4"
python main.py adaptive-tool-select "Ich brauche ein Tool fuer Aktienkurse der letzten 5 Jahre"
```

## API

```text
GET /api/cognitive/adaptive-tool-selection/status
GET /api/cognitive/adaptive-tool-selection/preview?query=...
```

## Sicherheit

- Keine Tool-Ausführung
- Keine Code-Generierung
- Kein Registry-Schreiben
- Tool-Gaps erfordern User Approval
- Profil-/Security-Policy bleibt deterministisch in Python
