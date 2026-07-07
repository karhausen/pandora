# MVP 30.10 – Core Triage Report

Status: **ANALYZE-MVP**  
Basis: `core_runtime_analysis_mvp30_9.json` / `core_runtime_analysis_mvp30_9.md`  
Erstellt: 2026-07-07T08:28:51.967114+00:00

## Ziel

Die 26 statischen Legacy-Kandidaten aus MVP 30.9 werden eingeordnet.  
Es wird **nichts gelöscht** und **nichts verschoben**.

## Ausgangslage aus MVP 30.9

- Python-Dateien gesamt: **292**
- Core-Module gesamt: **266**
- Statisch erreichbar: **181**
- Nicht erreichbar: **85**
- Legacy-Kandidaten, statisch/konservativ: **26**

## Kategorien

| Kategorie | Bedeutung | Aktion |
|---|---|---|
| A | Behalten | Nicht anfassen |
| B | Schützen / später prüfen | Nicht verschieben |
| C | Wahrscheinlich Legacy | Noch nicht verschieben |
| D | Kandidat für Legacy-Quarantäne | Nur nach kurzem grep + Smoke-Test verschieben |

## Ergebnis

| Kategorie | Anzahl |
|---|---:|
| A – Behalten | 3 |
| B – Schützen / später prüfen | 5 |
| C – Wahrscheinlich Legacy | 10 |
| D – Legacy-Quarantäne-Kandidat | 8 |

## Triage-Tabelle

| Datei | Kat. | Grund | Empfehlung |
|---|---:|---|---|
| `core/action_proposal_engine.py` | C | Nicht importiert; Proposal-System ist aktuell deaktiviert, aber viele Proposal-Endpunkte existieren noch. | Wahrscheinlich Legacy; erst verschieben, wenn Proposal-/Action-Pfade inventarisiert sind. |
| `core/capability_registry.py` | B | Capability-Thema ist deaktiviert, aber langfristig geplant. | Behalten bis Capability-System später wieder aktiviert/neu bewertet wird. |
| `core/capability_relationships.py` | B | Capability-Thema ist deaktiviert, aber langfristig geplant. | Behalten bis Capability-System später wieder aktiviert/neu bewertet wird. |
| `core/chat_response_router.py` | D | Alte Chat-Routing-Logik; Router darf nicht fachlich entscheiden. | Als ersten Legacy-Kandidaten in Quarantäne verschieben, nach kurzem grep. |
| `core/execution_context.py` | C | Nicht importiert; generische Klasse kann aber dynamisch oder konzeptionell vorgesehen sein. | Wahrscheinlich generische Altlast; vor Verschieben nach Namensreferenzen suchen. |
| `core/observation/detectors/capability_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/observation/detectors/gui_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/observation/detectors/memory_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/observation/detectors/review_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/observation/detectors/runtime_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/observation/detectors/tool_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/observation/detectors/workflow_detector.py` | D | Einzeldetektor ist statisch ungenutzt; Observation-System ist aktuell deaktiviert/nicht im Hauptpfad. | In Legacy-Quarantäne verschieben, wenn keine dynamische Detector-Registry existiert. |
| `core/obsidian_export.py` | A | Obsidian/Vault ist aktuelles Hauptziel; statische Nicht-Erreichbarkeit kann CLI-/Altpfade übersehen. | Behalten bis Vault-Pfad vollständig verifiziert ist. |
| `core/obsidian_indexer.py` | A | Obsidian/Vault ist aktuelles Hauptziel; statische Nicht-Erreichbarkeit kann CLI-/Altpfade übersehen. | Behalten bis Vault-Pfad vollständig verifiziert ist. |
| `core/obsidian_search.py` | A | Obsidian/Vault ist aktuelles Hauptziel; statische Nicht-Erreichbarkeit kann CLI-/Altpfade übersehen. | Behalten bis Vault-Pfad vollständig verifiziert ist. |
| `core/operations_issue_workflow.py` | C | Nicht importiert; Operations-Bereich existiert aber teilweise aktiv. | Wahrscheinlich Legacy; zuerst Operations-Endpunkte gegen aktuelles Modul prüfen. |
| `core/prioritization/evaluators/benefit.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/prioritization/evaluators/confidence.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/prioritization/evaluators/effort.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/prioritization/evaluators/frequency.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/prioritization/evaluators/risk.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/prioritization/evaluators/urgency.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/prioritization/evaluators/user_value.py` | C | Evaluator-Module sind wahrscheinlich Paketbestandteile; statische Analyse kann dynamische Ladewege übersehen. | Nicht verschieben; zuerst gesamtes Prioritization-Paket entscheiden. |
| `core/recovery.py` | B | Nicht importiert, aber potenziell sicherheits- oder betriebsrelevant. | Behalten oder separat prüfen; Querschnitts-/Sicherheits-/Betriebsmodul. |
| `core/resource_monitor.py` | B | Nicht importiert, aber potenziell sicherheits- oder betriebsrelevant. | Behalten oder separat prüfen; Querschnitts-/Sicherheits-/Betriebsmodul. |
| `core/security.py` | B | Nicht importiert, aber potenziell sicherheits- oder betriebsrelevant. | Behalten oder separat prüfen; Querschnitts-/Sicherheits-/Betriebsmodul. |

## Erste sichere Arbeitsmenge für MVP 31.0

Nur diese Dateien sind aktuell als **D** markiert. Auch diese werden nicht blind gelöscht, sondern maximal nach `legacy/` verschoben:

- `core/chat_response_router.py`
- `core/observation/detectors/capability_detector.py`
- `core/observation/detectors/gui_detector.py`
- `core/observation/detectors/memory_detector.py`
- `core/observation/detectors/review_detector.py`
- `core/observation/detectors/runtime_detector.py`
- `core/observation/detectors/tool_detector.py`
- `core/observation/detectors/workflow_detector.py`

## Nicht anfassen im aktuellen Zielbereich

Diese Obsidian-Dateien sind statisch unbenutzt, werden aber bewusst geschützt, weil Vault/Gedächtnis unser aktuelles Hauptziel ist:

- `core/obsidian_export.py`
- `core/obsidian_indexer.py`
- `core/obsidian_search.py`

## Empfehlung

1. MVP 30.10 als Analyse abschließen.
2. Vor MVP 31.0 einen lokalen `grep`/`ripgrep` gegen die D-Kandidaten laufen lassen.
3. Danach nur D-Kandidaten nach `legacy/` verschieben.
4. Nach jedem Move: API starten + Chat/Vault-Smoke-Test.
