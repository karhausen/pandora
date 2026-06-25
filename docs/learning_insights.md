# MVP 24.1 – Learning Insights

Learning Insights sind prüfbare Hinweise, die aus Learning Events, Metrics und Patterns entstehen.

Wichtig: Die Funktion bleibt observe-only. Pandora führt keine Tools aus, aktiviert keine Skills und ändert keine Knowledge-Dateien automatisch.

## CLI

```bash
python main.py learning-insight-status
python main.py learning-insights --rebuild
python main.py learning-insight-show <insight_id>
python main.py learning-insight-decide <insight_id> --decision reviewed
```

## GUI

Öffne `/learning` und klicke auf **Insights erzeugen**.

## Review Inbox

Learning Insights werden zusätzlich unter `proposals/learning_insights/` als `proposal.json` abgelegt. Dadurch erscheinen sie in der Review Inbox und in der Unified Action Inbox.
