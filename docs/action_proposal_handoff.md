# MVP 27.8 – Action Proposal Handoff

Der Action Proposal Handoff verbindet den Review-to-Action Workflow mit den bestehenden Tool-, Knowledge- und Core-Recommendation-Flows.

## Ziel

Wenn Pandora aus einem Review eine Aktion ableitet und der Benutzer `Vorschlag ausarbeiten` bestätigt, entsteht daraus ein passender, reviewbarer Proposal-Brief.

Der Handoff erzeugt **keinen Code**, schreibt **keine Dateien**, aktiviert **keine Tools** und verändert **keinen Core**.

## Ablauf

```text
Review Cycle
↓
Review-to-Action Karte
↓
User bestätigt: Vorschlag ausarbeiten
↓
Action Proposal Handoff
↓
Tool / Knowledge / Core Recommendation Workflow
↓
Reviewbarer Proposal-Brief
```

## CLI

```bash
python main.py action-proposal-handoff-status
python main.py action-proposal-handoff-preview "Pandora braucht ein Tool für Aktienhistorien"
```

Optionen:

```bash
--cadence weekly|monthly
--user-action ja|später|nein
--action-id <id>
--provider-name <provider>
--model <model>
--timeout 8.0
--max-items 8
```

## API

```text
GET /api/cognitive/action-proposal-handoff/status
GET /api/cognitive/action-proposal-handoff/preview?query=...
```

## Sicherheitsregeln

- keine Codegenerierung
- keine Tool-Ausführung
- keine Tool-Aktivierung
- keine Vault-/Knowledge-Schreiboperation
- keine Core-Änderung
- kein Release-Build
- immer Review/Freigabe erforderlich

## Bedeutung

MVP 27.8 macht den kognitiven Verbesserungsprozess durchgängiger: Reviews enden nicht mehr nur in Empfehlungen, sondern können kontrolliert in konkrete Proposal-Briefs übergeben werden.
