# MVP 20.2 – Tool Factory GUI Workflow

MVP 20.2 ergänzt die User-GUI um einen geführten Tool-Factory-Workflow.

## Ziel

Wenn Pandora aus einer Nutzerfrage einen neuen Tool-Vorschlag erstellt, soll der Nutzer den Proposal direkt in der GUI prüfen und kontrolliert weiterführen können:

```text
User fragt nach neuer Fähigkeit
↓
Capability Gap
↓
Tool Proposal
↓
GUI Workflow
↓
Approve
↓
Install
↓
Tool ist aktiv
```

## GUI-Funktionen

Die User-GUI enthält jetzt den Bereich **Tool Factory Workflow** mit:

- Proposal-Liste
- Proposal-Details
- Statusanzeige (`VALIDATED`, `APPROVED`, `INSTALLED`, `FAILED`, `REJECTED`)
- Buttons für `Approve`, `Install`, `Reject`
- automatische Anzeige, wenn eine Chat-Antwort ein `proposal_id` enthält

## Sicherheitsprinzip

Die GUI macht keine automatische Aktivierung. Der Nutzer muss weiterhin bewusst klicken:

```text
VALIDATED → Approve → APPROVED → Install → INSTALLED
```

## Verwendete API-Endpunkte

```text
GET  /tool-proposals
GET  /tool-proposals/{proposal_id}
POST /tool-proposals/{proposal_id}/approve
POST /tool-proposals/{proposal_id}/reject
POST /tool-proposals/{proposal_id}/install
```

## Manueller Test

1. API starten:

```bash
python3 main.py api
```

2. User-GUI öffnen:

```text
http://127.0.0.1:8000/
```

3. Anfrage stellen:

```text
Ich möchte Wörter zählen können
```

4. Im Tool Factory Workflow:

```text
Proposal prüfen → Approve → Install
```

5. Tool-Liste prüfen:

```bash
python3 main.py tool-list
```
