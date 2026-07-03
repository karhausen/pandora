# MVP 29.4.5 – Generic Tool Generator Architecture Fix

- ToolGenerator und ToolTestGenerator capability-agnostisch neu aufgebaut.
- `_looks_like_*`-Sonderfälle entfernt.
- ToolProposalManager nutzt ToolDesign → Code Generation statt domänenspezifischer Python-Zweige.
- Fallbacks erzeugen nur noch reviewpflichtige Scaffolds und keine fachlich vorgetäuschten Tools.

# MVP 29.4.1 – LLM Capability Gap Analyzer

- Added semantic LLM-first capability gap analysis against Pandora's current tool/skill/knowledge state.
- Removed keyword/pattern fallback as the primary capability-gap decision path.
- Prevented unsafe fallback to calculator for requests such as “Ich brauche ein Tool, das Prim-Zahlen berechnet.”
- Added `python main.py capability-gap analyze <task>` and `POST /api/capability-gap/analyze`.
- Added prime-number proposal validation path with deterministic SAFE generated code/tests.
- Maintains controlled evolution: recommendations/proposals only, no automatic activation.



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
