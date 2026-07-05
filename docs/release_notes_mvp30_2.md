# MVP 30.2 – Core Inventory & Cleanup

Basis: MVP 30.1 Unified Capability Model

## Ziel

Den `core`-Ordner stabilisieren und alte Keyword-/Regelpfade entschärfen, bevor neue Capability-Features gebaut werden.

## Änderungen

- `core/chat_response_router.py` ist jetzt nur noch ein deaktivierter Compatibility-Shim.
- `core/capability_detector.py` nutzt keine Keyword-Tabellen mehr und arbeitet nur noch mit strukturierter Analyse.
- `core/action_planner.py` nutzt keine Regel-/Keyword-Fallbacks mehr und führt Tools nur noch aus, wenn sie strukturiert vorgeschlagen wurden.
- `core/capability_analyzer.py` erzeugt keine Capability-Gaps mehr aus Request-Keywords.
- Neue Inventar-Doku: `docs/core_inventory_mvp30_2.md`.
- Neuer Regressionstest: `tests/test_core_cleanup_mvp30_2.py`.

## Nicht gemacht

- Keine Sidebar / Live-Console.
- Keine neuen Features.
- Keine Evolution-Erweiterung.
- Keine riskante Löschung großer Core-Bereiche.

## Test

`pytest -q`

Ergebnis: `8 passed`
