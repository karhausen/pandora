# MVP 24.9 – Review Scheduler & Manual Run Center

Der Review Scheduler ist eine kontrollierte Auslöseschicht für die Night Review Engine.

## Ziel

Pandora soll Night Reviews geplant oder manuell ausführen können, ohne einen versteckten Hintergrundprozess zu starten.

## Prinzip

- kein eigener Daemon
- keine automatische Ausführung von Actions
- externe Scheduler möglich: Cron, Windows Task Scheduler, Docker Cron
- GUI/CLI/API können einen Lauf auslösen
- Ergebnisse landen als Reports und reviewbare Actions in der Unified Action Inbox

## Konfiguration

In `.env`:

```env
PANDORA_REVIEW_SCHEDULER_ENABLED=false
PANDORA_NIGHT_REVIEW_TIME=02:00
PANDORA_NIGHT_REVIEW_LIMIT=200
PANDORA_NIGHT_REVIEW_CREATE_ACTIONS=true
```

## CLI

```bash
python main.py review-scheduler-status
python main.py review-scheduler-run --write --create-actions
python main.py review-scheduler-run-if-due
python main.py review-scheduler-run-if-due --force
python main.py review-scheduler-history
```

## Web

```text
/review-scheduler
```

## Sicherheit

Der Scheduler erzeugt nur Reports und reviewbare Empfehlungen. Er verändert keine Tools, Skills oder Core-Dateien und führt keine Actions automatisch aus.
