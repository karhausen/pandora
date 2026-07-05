# MVP 30.3 – Cognitive Reasoning Layer

## Ziel
Pandora soll vor jeder Tool-/Capability-Entscheidung zuerst das eigentliche Problem verstehen.

## Änderungen
- Neuer `core/cognitive_reasoning_layer.py`.
- `CapabilityOrchestrator.decide()` nutzt jetzt zuerst die Denkschicht.
- `CapabilitySnapshot` enthält zusätzlich `workflow:python_task_execution`.
- `workflow:tool_factory` ist explizit als letzter Schritt beschrieben.
- Fallback bei nicht verfügbarer LLM-Entscheidung bleibt sicher: Chat mit freigegebenem Kontext, keine Tool-Auswahl, keine Tool-Erstellung.

## Erwartetes Verhalten
- Projekt-/ToDo-Fragen: Wissen/Memory/Vault nutzen oder transparent sagen, wenn keine belastbare ToDo-Liste gefunden wurde.
- Einmalige Berechnungen: vorhandene Fähigkeiten/Python nutzen, nicht sofort Tool-Entwicklung.
- Echte dauerhafte neue Pandora-Fähigkeit: Proposal erstellen, aber erst wenn vorhandene Capabilities nicht ausreichen.

## Nicht enthalten
- Keine Sidebar/Live-Console.
- Keine neue GUI.
- Keine neue Tool Factory.
