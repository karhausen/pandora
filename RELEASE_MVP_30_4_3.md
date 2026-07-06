# MVP 30.4.3 – LLM Timeout & Light Cleanup

Ziel: MVP 30.4 stabilisieren, ohne Architekturänderung.

## Änderungen

- Route-Planner Timeout von 30s auf 90s erhöht.
- Chat-LLM Timeout von 30s auf 90s erhöht.
- Beide Timeouts sind per Environment überschreibbar:
  - `PANDORA_ROUTE_PLANNER_TIMEOUT`
  - `PANDORA_CHAT_LLM_TIMEOUT`
- `openai` und `company_llm` Default-Timeout in `config/llm/llm_config.json` auf 90s gesetzt.
- Release-ZIP bereinigt:
  - `.pytest_cache` entfernt
  - `core_before_cleanup/` entfernt

## Nicht geändert

- Router bleibt reiner Dispatcher.
- LLM wählt Route.
- Tools, Skills, Capability Gap und Tool Factory bleiben deaktiviert.
- Keine Keyword-Routing-Logik ergänzt.
