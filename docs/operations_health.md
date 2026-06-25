# MVP 24.11 – Operations Health & System Diagnostics

Operations Health ergänzt das Operations Cockpit um eine zentrale, read-only Systemdiagnose.

## Ziele

- Projektstruktur prüfen
- zentrale Services auf Antwortfähigkeit prüfen
- Registration Validation sichtbar machen
- Webrouten/HTML-Seiten prüfen
- konkrete nächste Schritte bei Fehlern anzeigen

## CLI

```bash
python main.py operations-health
python main.py operations-health-checks
```

## API

```text
GET /api/gui/operations-health/status
GET /api/gui/operations-health/checks
```

## GUI

```text
/operations-health
```

## Sicherheit

Operations Health ist read-only:

- keine Actions werden ausgeführt
- keine Tools oder Skills werden aktiviert
- keine Core-Dateien werden geändert
- keine Release-Artefakte werden geschrieben
