# MVP 26.5 – Cognitive Integration & Regression Hardening

## Ziel

MVP 26.5 verbindet die kognitiven Bausteine aus MVP 25.2 bis 26.4 zu einem nachvollziehbaren Preview- und Regression-Flow.

Der Fokus liegt nicht auf neuen Aktionen, sondern auf Stabilität:

- zentrale Entscheidung nachvollziehen
- Approval-Flow prüfen
- Proposal-Review prüfen
- Execution-Gate prüfen
- Vault-/Context-Regression absichern
- gefährliche Seiteneffekte vermeiden

## Grundregel

MVP 26.5 führt nichts aus.

Keine Tool-Ausführung.
Keine Codegenerierung.
Keine Knowledge-Writes.
Keine Tool-Aktivierung.
Keine Core-Änderung.
Keine Release-Erstellung.

## Neue Komponente

`core/cognitive_integration_regression.py`

Sie integriert:

- Cognitive Context Pipeline
- Central Decision Engine
- Approval Interaction Workflow
- Proposal Review Loop
- Proposal Execution Gate

## CLI

```bash
python main.py cognitive-integration-status
python main.py cognitive-integration-preview "Was war meine letzte Notiz?"
python main.py cognitive-integration-preview "Ich brauche ein Tool fuer Aktienkurse" --user-decision ja
python main.py cognitive-regression-run
```

## API

```text
GET /api/cognitive/integration/status
GET /api/cognitive/integration/preview?query=...
GET /api/cognitive/regression/run
```

## Regression-Szenarien

- `obsidian_last_note_context`
  - schützt den GUI-/Vault-Kontextpfad
  - darf nicht versehentlich als Tool- oder Core-Proposal enden

- `tool_gap_approval`
  - fehlendes Tool muss zu einer User-Frage führen
  - kein automatisches Bauen oder Aktivieren

- `knowledge_gap_approval`
  - Wissensänderungen müssen vorgeschlagen und geprüft werden

- `core_gap_approval`
  - Core-Verbesserungen brauchen Proposal, Tests, Audit und Freigabe

## User-Prinzip

Pandora soll den Benutzer nur an echten Entscheidungspunkten fragen:

```text
Wir brauchen Tool XY. Soll ich den Vorschlag ausarbeiten?
```

Danach:

```text
Hier ist der Vorschlag. Passt er oder soll ich nachbessern?
```

## Release-Regel

Dieser MVP ist ein Integrations- und Sicherheits-MVP. Er soll verhindern, dass neue Cognitive-Features bestehende Kernfunktionen wie Chat, Vault-Zugriff oder Freigabeprozesse beschädigen.
