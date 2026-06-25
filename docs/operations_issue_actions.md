# MVP 24.12 – Operations Issue Actions

Pandora kann erkannte Operations-Probleme jetzt in prüfbare Actions überführen.

## Prinzip

```text
Operations Health
  ↓
Operations Issue Detector
  ↓
Operations Issue Action
  ↓
Unified Action Inbox
  ↓
User entscheidet
```

## Sicherheitsregel

Pandora repariert nichts automatisch. Es werden nur JSON-Proposals unter
`proposals/operations_issue_actions/` erzeugt.

## CLI

```bash
python main.py operations-issues
python main.py operations-issue-scan
python main.py operations-issue-create-actions
python main.py operations-issue-show <id>
```

## GUI

`/operations-issues`

Zeigt Issues, erzeugte Actions und Verknüpfung zur Unified Action Inbox.
