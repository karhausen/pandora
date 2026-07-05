# MVP 30.4.2 – Vault Context Enforcement Fix

## Ziel
MVP 30.4 bleibt fokussiert auf genau zwei Pfade:

1. Vault/Knowledge → LLM
2. Direkt → LLM

Keine Tools, keine Tool-Factory, keine Capability-Gap-Entwicklung.

## Fixes

- Wenn der KnowledgeIntentRouter fälschlich `needs_knowledge=false` liefert, wird eine begrenzte, policy-sichere Retrieval-Validierung durchgeführt.
- Wenn diese Validierung relevante Vault-/Knowledge-Treffer findet, wird der Chat-Pfad auf `answer_with_context` korrigiert.
- Der finale LLM-Prompt weist ausdrücklich darauf hin, dass bereitgestellter Vault-/Knowledge-Kontext genutzt werden darf und dass das LLM nicht behaupten soll, keinen Zugriff zu haben.
- GUI-Session-API aus MVP 30.4.1 bleibt erhalten.

## Regression

- `Welche Test-Prompts habe ich?` darf nicht mehr ohne Vault-Kontext beantwortet werden, wenn relevante Knowledge-Treffer vorhanden sind.
- Direkte allgemeine Fragen bleiben möglich, wenn die Retrieval-Validierung keine relevanten Quellen findet.

## Tests

`26 passed`
