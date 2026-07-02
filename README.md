# Pandora Agent – MVP 28.7 Pattern Recognition Engine

Dieser Release baut auf MVP 28.6 auf und ergänzt eine Pattern Recognition Engine für die Self-Observation-Daten.

## Neu in MVP 28.7

- `core/pattern/` als eigenes Paket
- Pattern Schema für erkannte Muster
- Rule-based Pattern Detector
- Pattern Storage mit SQLite-Schicht
- Pattern Recognition Manager und Engine
- Detektoren für:
  - häufige Event-Typen
  - wiederkehrende Komponentenfehler
  - langsame Komponenten
  - wiederholte Capability Gaps
  - Review-Entscheidungsmuster
  - GUI-Nutzungsschwerpunkte
- CLI-Befehle für Pattern Recognition
- API-Endpunkte unter `/api/pattern/*`
- Maintenance-Link und neue Seite `/pattern`
- Genome-Konfiguration für Pattern Recognition

## Wichtige Grenze

MVP 28.7 erkennt Muster in vorhandenen Observation-Fakten.
Es erzeugt keine Proposals, aktiviert keine Änderungen und verändert keine Core-, Tool-, Skill- oder Genome-Dateien automatisch.

Der nächste logische Schritt ist MVP 28.8 – Improvement Prioritization.

## CLI Smoke Tests

```bash
python main.py pattern-status
python main.py pattern-health
python main.py pattern-statistics
python main.py pattern-detect --limit 500
python main.py pattern-detect --limit 500 --save
python main.py pattern-list --limit 20
```

## API

- `GET /api/pattern/status`
- `GET /api/pattern/health`
- `GET /api/pattern/detect`
- `GET /api/pattern/patterns`
- `GET /api/pattern/statistics`

## Release-Hinweis

Runtime-, Test- und Build-Artefakte sind nicht Teil der Clean-ZIP.
