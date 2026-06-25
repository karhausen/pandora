# MVP 24.8 – Night Review Engine

Die Night Review Engine prüft Pandora-Bereiche wie Unified Action Inbox, Workflows, Learning Patterns, Knowledge Governance, Capability Intelligence und Tool Improvements.

Sie erzeugt Reports und optionale reviewbare Empfehlungen für die Unified Action Inbox.

## Sicherheit

- keine automatische Ausführung
- keine Tool-Installation
- keine Skill-Aktivierung
- keine Core-Änderung
- nur Reports und reviewbare Actions

## CLI

```bash
python main.py night-review-status
python main.py night-review-run
python main.py night-review-run --no-write
python main.py night-review-reports
python main.py night-review-recommendations
```

## GUI

```text
/night-review
```
