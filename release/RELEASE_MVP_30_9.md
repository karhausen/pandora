# Release MVP 30.9 – Core Runtime Analysis

Status: ANALYZE-MVP

## Ziel

MVP 30.9 verändert keine produktive Core-Runtime. Der Zweck ist ein objektiver Überblick über den aktuellen Core-Zustand nach MVP 30.7/30.8.

## Enthalten

- `scripts/core_runtime_analyze.py`
- `docs/core_runtime_analysis_mvp30_9.md`
- `docs/core_runtime_analysis_mvp30_9.json`
- statische Importanalyse ab Entry Points `main` und `core.api`
- Liste statisch erreichbarer Core-Module
- Liste nicht erreichbarer Core-Module
- konservative Legacy-Kandidatenliste

## Wichtig

Die Analyse ist statisch. Dateien dürfen nicht allein aufgrund dieses Berichts gelöscht oder verschoben werden.

Nächster sinnvoller Schritt: Kandidaten manuell klassifizieren und erst danach einen separaten Cleanup-Build planen.
