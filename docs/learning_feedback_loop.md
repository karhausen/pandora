# MVP 24.2 – Learning Feedback Loop

Der Feedback Loop wandelt explizite User-Entscheidungen aus der Unified Action Inbox in beobachtbare Learning Events um.

Wichtig: Der Feedback Loop ist **observe-only**. Er führt keine Actions aus, installiert keine Tools, aktiviert keine Skills und verändert den Core nicht.

## CLI

```bash
python main.py learning-feedback-status
python main.py learning-feedback-collect --limit 1000
python main.py learning-feedback-report
python main.py learning-feedback-record <action_id> --decision reviewed --note "ok"
```

## API

```text
GET  /api/learning/feedback/status
POST /api/learning/feedback/collect
GET  /api/learning/feedback/report
POST /api/learning/feedback/{action_id}/record
```

## Zweck

- Akzeptierte Vorschläge werden als positives Signal gespeichert.
- Abgelehnte oder fehlerhafte Vorschläge werden als negatives Signal gespeichert.
- Zurückgestellte Vorschläge werden neutral bewertet.

Diese Signale verbessern spätere Learning Insights, lösen aber keine automatische Änderung aus.
