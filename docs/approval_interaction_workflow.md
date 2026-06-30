# MVP 26.2 – Approval Interaction Workflow

Der Approval Interaction Workflow sitzt direkt über der Central Decision Engine.
Er macht aus einer internen Decision eine einfache, benutzerfreundliche Freigabefrage.

## Ziel

Pandora soll nicht mit internen Details nerven.
Sie fragt nur an echten Entscheidungspunkten:

1. Soll ein Vorschlag ausgearbeitet werden?
2. Passt der Vorschlag oder soll nachgebessert werden?

## Ablauf

```text
User Request
↓
Central Decision Engine
↓
Approval Interaction Workflow
↓
Einfache Frage an den User
↓
Ja / Nein / Details / Nachbessern
↓
Kontrollierter Handoff an Tool-, Knowledge- oder Core-Review
```

## Sicherheitsgrenzen

Der Workflow erzeugt keinen Code, aktiviert keine Tools, schreibt keine Dateien und verändert keinen Core.
Er erzeugt nur einen kontrollierten Handoff.

## CLI

```bash
python main.py approval-interaction-status
python main.py approval-interaction-preview "Baue ein Tool für historische Aktienkurse"
python main.py approval-interaction-preview "Baue ein Tool für historische Aktienkurse" --user-decision ja
```

## API

```text
GET /api/cognitive/approval-interaction/status
GET /api/cognitive/approval-interaction/preview?query=...
```
