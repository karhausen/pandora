# MVP 27.6 – Cognitive Dashboard Integration

## Ziel

Das Cognitive Dashboard bündelt Pandoras kognitive Bausteine in einer zentralen, lesbaren Übersicht.
Es ersetzt keine bestehende Engine und erzeugt keine parallele Entscheidungslogik.

## Enthaltene Quellen

- Central Decision Engine
- GUI Decision Inbox
- Goal Manager
- Priority Engine
- Review Cycle Engine
- Working Memory

## Sicherheitsregel

Das Dashboard ist strikt read-only.

Es darf nicht:

- Tools ausführen
- Tools aktivieren
- Dateien schreiben
- Obsidian/Vault ändern
- Knowledge Base ändern
- Memory persistieren
- Core-Code verändern

## CLI

```bash
python main.py cognitive-dashboard-status
python main.py cognitive-dashboard-preview "Prüfe den aktuellen Stand von Pandora"
python main.py cognitive-dashboard-preview "Monatsreview für Pandora" --cadence monthly
```

## API

```text
GET /api/cognitive/dashboard/status
GET /api/cognitive/dashboard/preview?query=...
```

## GUI

```text
/cognitive-dashboard
```

## Zweck

Das Dashboard macht sichtbar:

- aktuelle zentrale Entscheidung
- offene Freigabepunkte
- Zielvorschläge
- Prioritäten
- Review-Fokus
- Working-Memory-Snapshot
- vollständigen Trace für Debugging und Regression
