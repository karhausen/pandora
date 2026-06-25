# MVP 24.4 – Learning Pattern Actions

Learning Pattern Actions wandeln erkannte Learning Patterns in prüfbare Actions um.

Wichtig: Diese Komponente führt keine Änderungen aus. Sie erzeugt nur Vorschläge, die in der Unified Action Inbox sichtbar werden.

## CLI

```bash
python main.py learning-pattern-action-status
python main.py learning-pattern-actions --rebuild
python main.py learning-pattern-actions --rebuild --rebuild-patterns
python main.py learning-pattern-action-show <action_id>
python main.py learning-pattern-action-decide <action_id> --decision reviewed
```

## API

```text
GET  /api/learning/pattern-actions/status
GET  /api/learning/pattern-actions
POST /api/learning/pattern-actions/rebuild
GET  /api/learning/pattern-actions/{action_id}
POST /api/learning/pattern-actions/{action_id}/decision
```

## Workflow

```text
Learning Events
↓
Learning Pattern Detection
↓
Learning Pattern Actions
↓
Unified Action Inbox
↓
User Review
```

## Sicherheitsregeln

- keine automatische Tool-Installation
- keine Skill-Aktivierung
- keine Core-Änderung
- keine automatische Ausführung
- User-Freigabe bleibt Pflicht
