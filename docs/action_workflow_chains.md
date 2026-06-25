# MVP 24.6 – Action Workflow Chains

Pandora arbeitet Actions jetzt als kontrollierte Workflow-Ketten ab.

## Prinzip

`accepted_for_next_step` führt **nichts automatisch aus**. Stattdessen wird die aktuelle Action als erledigt betrachtet und eine sichere Folge-Action erzeugt.

## Beispiel

1. Vorschlag prüfen
2. Ausführungsplan prüfen
3. Ausführung bestätigen
4. Ergebnis prüfen

Fehlerhafte Schritte bleiben in der Unified Action Inbox sichtbar und blockieren den Workflow.

## CLI

```bash
python main.py workflow-status
python main.py workflow-list
python main.py workflow-show WF-...
python main.py action-inbox-decide <id> --decision accepted_for_next_step
```
