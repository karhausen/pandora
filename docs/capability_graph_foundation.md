# MVP 23.0 – Capability Graph Foundation

Der Capability Graph verbindet Pandoras Fähigkeiten mit vorhandenen Tools, Skills, Knowledge-Dokumenten und offenen Capability Gaps.

## Ziel

Pandora soll nicht nur Dateien und Tools kennen, sondern Beziehungen verstehen:

- Welche Fähigkeit wird durch welches Tool unterstützt?
- Welche Skills gehören zu einer Fähigkeit?
- Welche Knowledge-Dokumente beschreiben ein Thema?
- Wo sind noch offene Capability Gaps?

## Datenquellen

- Tool Registry
- Skill Registry
- User Knowledge Base
- Knowledge Metadata und Tags
- Capability Gap Reports

## Speicherung

```text
data/capability_graph/
├── graph.json
├── nodes.json
└── edges.json
```

Diese Dateien sind generierte Daten. Sie dürfen neu aufgebaut werden.

## CLI

```bash
python main.py capability-status
python main.py capability-rebuild
python main.py capability-list
python main.py capability-show <name-or-id>
```

## API

```text
GET  /api/capabilities
GET  /api/capabilities/{capability}
GET  /api/capabilities/graph
POST /api/capabilities/rebuild
```

## Regeln

- Der Graph ist zunächst read-only gegenüber Tools, Skills und Knowledge.
- Rebuild erzeugt nur Graph-Dateien.
- Es werden keine Tools, Skills oder Knowledge-Dateien geändert.
- Private Knowledge bleibt private; der Graph speichert nur Metadaten und Dateipfade, keine geheimen Inhalte.
