# MVP 24.10 – Operations Cockpit Cleanup

Pandora hatte inzwischen mehrere Operations-Seiten: Action Inbox, Workflow Dashboard, Night Review, Review Scheduler und Classic Operations. MVP 24.10 bündelt diese Sicht in einem zentralen Cockpit.

## Ziele

- weniger Seitensprünge für den User
- klare Priorisierung: Fehler, blockierte Workflows, offene Actions
- schnelle Links zu den Detailseiten
- Night Review Preview und Scheduler Manual Run aus einer Stelle
- keine automatische Ausführung von Actions

## Neue Seite

```text
/operations-cockpit
```

## Neue API

```text
GET  /api/gui/operations-cockpit/dashboard
POST /api/gui/operations-cockpit/night-review-preview
POST /api/gui/operations-cockpit/scheduler-run
```

## Neue CLI

```bash
python main.py operations-cockpit
python main.py operations-cockpit-night-preview --limit 200
python main.py operations-cockpit-scheduler-run --limit 200
```

## Sicherheitsregeln

Das Cockpit ist eine Leitstelle, kein Autopilot:

- keine Core-Änderungen
- keine Tool-/Skill-Aktivierung
- keine Action-Ausführung
- manuelle Review-Runs erzeugen nur prüfbare Actions
