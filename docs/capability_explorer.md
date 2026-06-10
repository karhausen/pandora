# Capability Explorer

Der Capability Explorer ist die erste GUI für den Capability Graph.

## Zweck

Pandora zeigt damit Fähigkeiten und ihre Beziehungen zu folgenden Quellen:

- Tools
- Skills
- Knowledge-Dokumente
- Capability Gaps

## Start

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Danach öffnen:

```text
http://127.0.0.1:8000/capability-explorer
```

## Funktionen

- Capability Graph neu aufbauen
- Capabilities durchsuchen
- Capability-Details anzeigen
- verknüpfte Tools, Skills, Knowledge-Dokumente und Gaps sehen
- Raw Graph Detail prüfen

## Sicherheitsprinzip

Der Explorer ist read-only. Der Button „Graph neu aufbauen“ liest bestehende Quellen und schreibt nur die Graph-Dateien unter `data/capability_graph/` neu. Tools, Skills und Knowledge-Dateien werden nicht verändert.
