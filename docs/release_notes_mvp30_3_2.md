# MVP 30.3.2 – Decision Enforcement & Context Guard

## Ziel

Der validierte Capability-Entscheid muss den tatsächlichen Antwortpfad bestimmen. Alte Wissensnotizen oder Testpläne dürfen eine Tool-/Capability-Entscheidung nicht mehr übersteuern.

## Problem aus dem Primzahlen-Test

Die Reasoning-/Analyse-Antwort kam teilweise noch im alten Schema mit `next_action` und `suggested_tools` zurück. Dadurch fiel Pandora auf `answer_with_context` zurück, lud Obsidian-Kontext und bekam den alten Primzahlen-Testplan in den Prompt. Das LLM antwortete daraufhin wieder mit generischen Tool-Proposal-Schritten.

## Änderungen

- `CapabilityOrchestrator` normalisiert ältere strukturierte LLM-Schemas:
  - `next_action: use_tool` wird zu `action: use_tool`
  - `suggested_tools[0]` wird zu `requested_tool`
  - `required_capabilities` wird zu `needed_capabilities`
- `ChatService` enthält einen Context Guard:
  - kein Knowledge-/Obsidian-Kontext für `planner_worker`
  - kein Knowledge-/Obsidian-Kontext für `clarify`
  - kein Knowledge-/Obsidian-Kontext für `answer_directly`
  - Kontext nur noch bei explizitem Kontextbedarf
- Damit können alte Vault-Testpläne die aktive Entscheidung nicht mehr in Richtung Proposal verschieben.

## Erwartetes Verhalten beim Primzahlen-Test

Wenn die Reasoning-Schicht ein vorhandenes Werkzeug/Workflow als ausreichend sieht, läuft Pandora nicht mehr in freien Chat mit Obsidian-Kontext. Wenn keine ausführbaren Eingaben vorhanden sind, muss Pandora nach den fehlenden Daten bzw. nach der Bestätigung für ein dauerhaftes Tool fragen.

## Tests

`17 passed`
