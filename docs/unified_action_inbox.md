# MVP 23.6 – Unified Action Inbox

Die Unified Action Inbox ist die zentrale Arbeitsliste für Pandora. Sie fasst offene Aufgaben aus Review Inbox, Capability Actions, Obsidian Import Candidates, Tool-/Skill-Vorschlägen, Night Reports und Maintenance Reports zusammen.

## Ziel

Der User soll nicht mehr auf vielen Seiten suchen müssen, sondern morgens eine zentrale Inbox öffnen:

- Was ist offen?
- Was ist fehlgeschlagen?
- Was wurde erledigt?
- Welche Details, Logs und Artefakte gehören zu einer Action?

## GUI

Neue Seite:

```text
/action-inbox
```

Oben: offene Actions.  
Darunter: erledigte Actions.

Fehlerhafte Actions bleiben oben und werden markiert.

## Detailansicht

```text
/action-inbox/<id>
```

Zeigt:

- Zusammenfassung
- Begründung
- geplante Aktion
- Logs
- Fehler
- Artefakte
- Review-State
- Rohinhalt

## CLI

```bash
python main.py action-inbox-status
python main.py action-inbox-list
python main.py action-inbox-list --include-done
python main.py action-inbox-show <action_id>
python main.py action-inbox-decide <action_id> --decision reviewed --note "geprüft"
```

## API

```text
GET  /api/actions/dashboard
GET  /api/actions
GET  /api/actions/{action_id}
POST /api/actions/{action_id}/decision
```

## Sicherheitsregeln

Die Inbox führt keine Actions aus. Entscheidungen schreiben nur `review_state.json` neben die jeweilige Quelle. Tool-Installation, Core-Änderungen oder Importe bleiben in ihren spezialisierten Workflows.
