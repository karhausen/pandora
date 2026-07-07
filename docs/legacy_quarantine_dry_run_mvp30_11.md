# MVP 30.11 – Legacy Quarantine Dry Run

Status: **ANALYZE-MVP / DRY RUN**. Es wurden keine Core-Dateien verschoben, gelöscht oder geändert.

## Grundlage

- Quelle 1: `core_runtime_analysis_mvp30_9.json`
- Quelle 2: `reports/core_triage_report_mvp30_10.json`
- Simuliert werden nur Kategorie-D-Kandidaten aus MVP 30.10.

## Ergebnis

- Simulierte Kandidaten: **8**
- Low Risk: **8**
- Medium Risk: **0**
- High Risk: **0**

## Dry-Run-Tabelle

| Datei | Zielpfad bei Quarantäne | statisches Importbruch-Risiko | importiert von |
|---|---:|---:|---|
| `core/chat_response_router.py` | `legacy/core/chat_response_router.py` | low | — |
| `core/observation/detectors/capability_detector.py` | `legacy/core/observation/detectors/capability_detector.py` | low | — |
| `core/observation/detectors/gui_detector.py` | `legacy/core/observation/detectors/gui_detector.py` | low | — |
| `core/observation/detectors/memory_detector.py` | `legacy/core/observation/detectors/memory_detector.py` | low | — |
| `core/observation/detectors/review_detector.py` | `legacy/core/observation/detectors/review_detector.py` | low | — |
| `core/observation/detectors/runtime_detector.py` | `legacy/core/observation/detectors/runtime_detector.py` | low | — |
| `core/observation/detectors/tool_detector.py` | `legacy/core/observation/detectors/tool_detector.py` | low | — |
| `core/observation/detectors/workflow_detector.py` | `legacy/core/observation/detectors/workflow_detector.py` | low | — |

## Bewertung

Alle 8 Kategorie-D-Kandidaten haben im statischen Graphen **kein importierendes Modul**. Das spricht für geringes Risiko bei einer späteren Quarantäne.

Trotzdem gilt: Diese Analyse beweist nicht, dass keine dynamischen Imports, CLI-Strings oder manuelle Pfadzugriffe existieren. Vor einem echten Move sollte lokal zusätzlich `grep`/IDE-Suche nach Modulnamen und Dateinamen laufen.

## Empfehlung

Nächster Schritt wäre **MVP 31.0 – Legacy Quarantine Move**, aber nur für diese 8 Low-Risk-Kandidaten und weiterhin mit Rückfallmöglichkeit über Git.
