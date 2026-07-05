# Release Notes – MVP 30.4 Knowledge Routing Stabilization

## Fokus

Vault/Gedächtnis und normale LLM-Interaktion stabilisieren.

## Geändert

- Neuer `KnowledgeIntentRouter`.
- `ChatService.run()` ist für diesen MVP auf zwei Wege reduziert:
  - Knowledge/Vault → LLM
  - Direkt → LLM
- Tool-Ausführung, Tool-Entwicklung, Planner/Worker und Capability-Gap sind im Chat-Hauptpfad deaktiviert.

## Tests

- Knowledge-Intent True lädt Vault-Kontext vor der LLM-Antwort.
- Knowledge-Intent False ruft direkt das LLM ohne Vault-Kontext.
- ChatService enthält keine Tool-/Gap-Ausführung im MVP-30.4-Hauptpfad.
