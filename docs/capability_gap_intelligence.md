# MVP 23.2 – Capability Gap Intelligence

Capability Gap Intelligence priorisiert offene Fähigkeitslücken aus dem Capability Graph.

## Ziel

Pandora soll nicht nur Beziehungen anzeigen, sondern erkennen, welche Fähigkeiten als Nächstes Aufmerksamkeit brauchen.

## Bewertungslogik

Die Analyse ist absichtlich deterministisch und sicher:

- vorhandener Capability Gap erhöht Priorität
- Wissen ohne Tool deutet auf fehlende Automatisierung hin
- Wissen ohne Skill deutet auf fehlende wiederverwendbare Fähigkeit hin
- Tool ohne Skill deutet auf fehlenden Workflow hin
- Gap ohne Knowledge deutet auf fehlende Dokumentation hin

## Sicherheit

Die Pipeline ist read-only:

- keine Tool-Erzeugung
- keine Skill-Aktivierung
- keine Core-Änderung
- nächste Schritte brauchen Review und Approval

## CLI

```bash
python main.py capability-intelligence
python main.py capability-intelligence --rebuild --limit 25
```

## API

```text
GET /api/capabilities/intelligence
POST /api/capabilities/intelligence/rebuild
```

## GUI

Der Capability Explorer zeigt einen Intelligence-Bereich mit Top-Lücken, Schweregrad, Gründen und empfohlenem nächsten Schritt.
