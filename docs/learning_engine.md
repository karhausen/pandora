# MVP 24.0 – Learning Engine Foundation

Die Learning Engine ist in MVP 24.0 bewusst **observe-only**.

Sie sammelt Ereignisse aus bestehenden Pandora-Workflows, berechnet einfache Metriken und schreibt keine Tools, Skills oder Core-Dateien.

## CLI

```bash
python main.py learning-status
python main.py learning-collect
python main.py learning-rebuild
python main.py learning-metrics
python main.py learning-patterns
python main.py learning-events-v24
```

## API

```text
GET  /api/learning/status
GET  /api/learning/metrics
GET  /api/learning/patterns
GET  /api/learning/events
POST /api/learning/collect
POST /api/learning/rebuild
```

## GUI

```text
/learning
```

## Speicher

```text
data/learning/events.jsonl
data/learning/metrics.json
data/learning/patterns.json
```

## Sicherheitsregel

Die Learning Engine beobachtet nur. Automatische Änderungen, Installationen oder Core-Anpassungen sind nicht erlaubt.
