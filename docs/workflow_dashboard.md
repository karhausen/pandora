# MVP 24.7 – Workflow Dashboard

Das Workflow Dashboard ist die zentrale Übersicht für Action Workflow Chains.

## Ziel

Der User soll nicht mehr zwischen Action Inbox, Import Review, Capability Explorer und Learning-Seiten springen müssen, um zu sehen, welche Workflows offen, blockiert oder abgeschlossen sind.

## GUI

Neue Seite:

```text
/workflow-dashboard
```

Sie zeigt:

- aktive Workflows
- blockierte Workflows
- abgeschlossene Workflows
- Fortschritt je Workflow
- aktuellen Schritt
- nächste User-Aktion
- Timeline je Workflow

## Sicherheit

Das Dashboard ist read-only:

- keine automatische Ausführung
- keine Tool-Installation
- keine Core-Änderung
- keine automatische Fortsetzung

Entscheidungen erfolgen weiterhin über die Action Inbox.

## CLI

```bash
python main.py workflow-dashboard-status
python main.py workflow-dashboard-list
python main.py workflow-dashboard-show <workflow_id>
```

## API

```text
GET /api/workflow-dashboard/status
GET /api/workflow-dashboard
GET /api/workflow-dashboard/workflows
GET /api/workflow-dashboard/workflows/{workflow_id}
```
