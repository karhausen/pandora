# MVP 29.7.2 – Semantic Capability Decision Engine

## Ziel

Die Capability-Entscheidung wird verbindlich semantisch getroffen:

```
User-Aufgabe
+ Capability Snapshot
  - Tools
  - Skills
  - Knowledge
  - Workflows
  - Capabilities
  - Genome
↓
LLM
↓
Python Validator
↓
Ausführen oder Capability Gap / Proposal
```

## Änderungen

- Neue `core/capability_snapshot.py`
- `core/capability_gap_analyzer.py` neu als `SemanticCapabilityDecisionEngine`
- Backwards-kompatibler Alias `LLMCapabilityGapAnalyzer`
- `ToolDevelopmentAgent` verwendet die neue Engine
- `tool_selection` hat keinen Mock-Fallback mehr in der LLM-Konfiguration
- Runtime-Mock-Antworten werden als nicht autoritativ zurückgewiesen
- Keine Keyword-/Pattern-Entscheidung im Hauptpfad
- Keine Capability-spezifischen Python-Entscheidungen
- Keine `_looks_like_*`-Logik im Tool-/Capability-Generator-Pfad

## Guardrails

- Ein generisches Tool wie `calculator` darf eine spezialisierte Capability nicht verdecken.
- Wenn die LLM-Analyse nicht verfügbar ist, wird kein unpassendes Fallback-Tool ausgeführt.
- Python validiert nur, ob ein vom LLM vorgeschlagenes vorhandenes Tool tatsächlich in der Registry existiert und die angefragte Capability in seiner Metadata beschreibt.

## Tests

- `python main.py selftest cli`
- `python main.py selftest api`
- `python main.py selftest integration`
- Guardrail-Test: inkonsistente LLM-Entscheidung wird als Gap behandelt
- Guardrail-Test: Calculator-Overmatch wird blockiert
- Guardrail-Test: Mock-Fallback wird zurückgewiesen
