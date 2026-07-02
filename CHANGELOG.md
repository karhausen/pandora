

## MVP 29.3 – Knowledge Evolution

- Neues Paket `core/knowledge_evolution` fuer Knowledge Health, Gap Detection, Freshness Checks und Proposal-Kandidaten.
- Neue CLI-Aliase `python main.py knowledge-evolution ...`.
- Neue API-Endpunkte `/api/knowledge-evolution/*`.
- Neue Maintenance-Seite `/knowledge-evolution`.
- Integration in Selftests und Maintenance Center.
- Keine automatische Aenderung von Knowledge-Dateien; alle Verbesserungen laufen ueber Proposal/Review.

## MVP 29.4 – Tool Evolution

- Neues Paket `core/tool_evolution` fuer Tool Health, Lifecycle-Uebersicht, Reviews und Refactoring-Kandidaten.
- Neue CLI-Aliase `python main.py tool-evolution ...` und Kompatibilitaet fuer `python main.py tools health/review/lifecycle`.
- Neue API-Endpunkte `/api/tool-evolution/*`.
- Neue Maintenance-Seite `/tool-evolution`.
- Integration in CLI/API/Integration-Selftests und Maintenance Center.
- Keine automatische Aenderung oder Aktivierung von Tool-Code; alle Verbesserungen laufen ueber Proposal/Review/User-Freigabe.
