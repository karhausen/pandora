# MVP 25.0 – Guided Self-Improvement Foundation

Guided Self-Improvement erzeugt kontrollierte Verbesserungsvorschläge aus vorhandenen Pandora-Signalen.

Quellen:

- Operations Issues
- Learning Pattern Actions
- Capability Actions
- Night Review Recommendations
- Unified Action Inbox Backlog

Wichtig: Diese Komponente führt nichts automatisch aus. Sie schreibt ausschließlich prüfbare Vorschläge nach `proposals/guided_self_improvement/`, die über Unified Action Inbox und Workflow Chains weiterbearbeitet werden.

## CLI

```bash
python main.py guided-improvement-status
python main.py guided-improvements --rebuild
python main.py guided-improvement-show <id>
python main.py guided-improvement-decide <id> --decision accepted_for_next_step
```

## GUI

```text
/guided-improvement
```

## Sicherheit

- keine Core-Änderung
- keine Tool-Installation
- keine Skill-Aktivierung
- keine automatische Ausführung
- nur reviewbare Vorschläge
