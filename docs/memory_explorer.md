# MVP 22.4 – Memory Explorer

Der Memory Explorer ist eine read-only GUI/API für Pandora Memory- und Proposal-Artefakte.

## Ziele

- Memory-Bereiche sichtbar machen
- JSON/JSONL-Dateien sicher previewen
- einfache Suche über Memory und Proposals anbieten
- User-GUI um einen klaren Einstieg erweitern

## GUI

Start API:

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Dann öffnen:

```text
http://127.0.0.1:8000/memory-explorer
```

## API

```text
GET /api/gui/memory/dashboard
GET /api/gui/memory/areas
GET /api/gui/memory/areas/{area}
GET /api/gui/memory/areas/{area}/files/{relative_path}
GET /api/gui/memory/search?query=...
```

Erlaubte Bereiche:

```text
memory
proposals
```

## CLI

```bash
python main.py memory-explorer-dashboard
python main.py memory-explorer-areas
python main.py memory-explorer-area memory
python main.py memory-explorer-show memory conversation_memory.json
python main.py memory-explorer-search error
```

## Sicherheitsgrenze

Der Explorer schreibt nicht in Memory-Dateien. Pfade werden auf erlaubte Bereiche begrenzt, damit kein Zugriff außerhalb von `memory/` oder `proposals/` möglich ist.
