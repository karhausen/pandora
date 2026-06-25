# MVP 22.5 – Night Mode Dashboard

Das Night Mode Dashboard macht Pandoras geplanten Nachtmodus sichtbar.

Ziele:

- Nachtberichte anzeigen
- Maintenance Reports anzeigen
- Capability-Gap-, Tool-Improvement- und Skill-/Review-Signale sichtbar machen
- Wartung als Dry Run prüfen
- keine automatischen Core-Änderungen
- keine automatische Tool-Installation
- keine automatische Skill-Aktivierung

## Web

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Dann öffnen:

```text
http://127.0.0.1:8000/night-mode
```

## API

```text
GET  /api/gui/night-mode/dashboard
GET  /api/gui/night-mode/reports
GET  /api/gui/night-mode/reports/{report_id}
POST /api/gui/night-mode/maintenance/preview
```

## CLI

```bash
python main.py night-mode-dashboard
python main.py night-mode-reports
python main.py night-mode-preview
```

## Sicherheitsregel

Der Night Mode ist bewusst observe-only. Er darf Vorschläge und Reports erzeugen, aber keine Tools installieren, keine Skills aktivieren und keine Core-Dateien ändern.
