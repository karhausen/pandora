# MVP 27.0 – Cognitive Planning Engine

Die Cognitive Planning Engine erzeugt vor Antwort oder Aktion einen prüfbaren Plan.

Sie beantwortet intern nicht direkt die Benutzerfrage, sondern beschreibt:

- welche Absicht erkannt wurde,
- welche Kontextquellen benötigt werden,
- welche Tools oder Skills relevant sind,
- welche Capability Gaps vorliegen,
- welche Reihenfolge sinnvoll ist,
- wo Benutzerfreigaben nötig sind.

## Sicherheitsgrenzen

Die Engine führt nichts aus.

Sie:

- liest keine Dateien,
- führt keine Tools aus,
- generiert keinen aktiven Tool-Code,
- schreibt nichts in Knowledge oder Obsidian,
- aktiviert keine Tools,
- verändert keinen Core.

## Prinzip

```text
User Request
↓
Request Interpreter
↓
Central Decision Engine
↓
Cognitive Planning Engine
↓
Reviewbarer Plan
↓
Python validiert spätere Schritte
```

## CLI

```bash
python main.py cognitive-planning-status
python main.py cognitive-plan "Was war meine letzte Notiz?"
```

## API

```text
GET /api/cognitive/planning/status
GET /api/cognitive/planning/preview?query=...
```
