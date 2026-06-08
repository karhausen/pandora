# MVP 21.9 – Minimal Web GUI

Pandora enthält ab MVP 21.9 eine einfache Web-Oberfläche für den Approval Workflow.

## Ziel

Die GUI macht die Review Inbox alltagstauglich:

- Dashboard für offene Vorschläge
- Review Inbox mit Risiko-Badges
- Detailansicht je Vorschlag
- Entscheidungsbuttons
- Audit-Anzeige

## Start

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Dann öffnen:

```text
http://127.0.0.1:8000/approval
```

## Sicherheitsgrenze

Die GUI ist absichtlich beschränkt:

- sie speichert Entscheidungen
- sie zeigt Audit-Daten
- sie startet keine Tool-Aktivierung
- sie installiert keine Skills
- sie verändert keinen Core-Code

Alle Entscheidungen gehen über die bestehende GUI Approval API:

```text
/api/gui/approval/*
```

Die eigentliche Umsetzung eines genehmigten Vorschlags bleibt ein separater kontrollierter Backend-Workflow.
