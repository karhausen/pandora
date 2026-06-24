# MVP 24.3 – Learning Pattern Detection

Pandora erkennt wiederkehrende Muster aus Learning Events und User-Feedback. Die Komponente ist bewusst **observe-only**: Sie erzeugt Hinweise, führt aber keine Tools aus, aktiviert keine Skills und verändert keinen Core-Code.

## CLI

```bash
python main.py learning-pattern-status
python main.py learning-patterns-detect --rebuild
python main.py learning-patterns-detect --include-reviewed
python main.py learning-pattern-show <pattern_id>
python main.py learning-pattern-decide <pattern_id> --decision reviewed
```

## API

```text
GET  /api/learning/pattern-detection/status
GET  /api/learning/pattern-detection
POST /api/learning/pattern-detection/rebuild
GET  /api/learning/pattern-detection/{pattern_id}
POST /api/learning/pattern-detection/{pattern_id}/decision
```

## Erkannte Muster

- wiederkehrende Event-/Result-Kombinationen
- Bereiche mit vielen Events
- offene Action-Backlogs
- viele negative oder positive Feedback-Signale

## Sicherheitsregeln

- keine automatische Ausführung
- keine Tool-Installation
- keine Skill-Aktivierung
- keine Core-Änderung
- jede Folgemaßnahme bleibt reviewpflichtig
