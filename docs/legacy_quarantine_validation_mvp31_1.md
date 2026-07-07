# MVP 31.1 – Legacy Quarantine Validation

Status: **VALIDATE-MVP**. Keine neuen Features, kein weiterer Legacy-Umzug.

## Ziel

Den Stand aus MVP 31.0 nach dem Legacy-Umzug erneut prüfen.

## Ergebnis

| Kennzahl | Vorher | Nachher | Delta |
|---|---:|---:|---:|
| Python-Dateien gesamt | 292 | 292 | 0 |
| Core-Module | 266 | 258 | -8 |
| Statisch erreichbar | 181 | 181 | 0 |
| Nicht erreichbar | 85 | 77 | -8 |
| Legacy-Kandidaten | 26 | 18 | -8 |

## Quarantäne-Prüfung

Alle 8 Kategorie-D-Dateien aus MVP 31.0 liegen nicht mehr unter `core/`, sondern unter `legacy/core/`.

```text
core/chat_response_router.py: core=NEIN, legacy=JA
core/observation/detectors/capability_detector.py: core=NEIN, legacy=JA
core/observation/detectors/gui_detector.py: core=NEIN, legacy=JA
core/observation/detectors/memory_detector.py: core=NEIN, legacy=JA
core/observation/detectors/review_detector.py: core=NEIN, legacy=JA
core/observation/detectors/runtime_detector.py: core=NEIN, legacy=JA
core/observation/detectors/tool_detector.py: core=NEIN, legacy=JA
core/observation/detectors/workflow_detector.py: core=NEIN, legacy=JA
```

## Tests

```text
36 passed
```

## Bewertung

Der Legacy-Umzug aus MVP 31.0 ist validiert. Die statische Analyse zeigt den erwarteten Rückgang um 8 Core-Module, 8 nicht erreichbare Module und 8 Legacy-Kandidaten.

## Regeln bleiben aktiv

- Router bleibt Dispatcher.
- Tools bleiben im Chat-Hauptpfad deaktiviert.
- Skills bleiben deaktiviert.
- Capability-Gap bleibt deaktiviert.
- Evolution bleibt deaktiviert.
