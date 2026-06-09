# MVP 22.2 – Tool Center GUI

Der Tool Center ist die erste Bedienoberfläche für installierte Pandora-Tools.

## Ziele

- installierte Tools sichtbar machen
- Tool-Status und Sicherheitsstufe zeigen
- Tool-Statistiken anzeigen
- sichere Lifecycle-Aktionen auslösen

## Web-GUI

Start:

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Öffnen:

```text
http://127.0.0.1:8000/tools-center
```

Die User-GUI `/` enthält ebenfalls einen Badge-Link zum Tool Center.

## API

```text
GET  /api/gui/tools/dashboard
GET  /api/gui/tools
GET  /api/gui/tools/{tool_id}
POST /api/gui/tools/{tool_id}/action
GET  /api/gui/tools/{tool_id}/stats
```

Erlaubte Aktionen:

```text
enable
disable
deprecate
```

Bewusst nicht enthalten: automatisches Löschen/Uninstall über die GUI. Das bleibt vorerst CLI/Backend-only, weil es destruktiver ist.

## Sicherheitslinie

Der Tool Center verändert keinen Code. Er ändert nur Lifecycle-Status im Registry-System.
