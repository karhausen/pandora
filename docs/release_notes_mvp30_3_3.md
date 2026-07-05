# MVP 30.3.3 – Knowledge Intent & Tool Contract Guard

Basis: MVP 30.3.2

## Ziel

Zwei Regressionen aus echten Tests beheben:

1. Fragen nach eigenen Test-Prompts/TODOs/Projektwissen dürfen nicht direkt aus Allgemeinwissen beantwortet werden.
2. Der Calculator darf keine freie natürliche Sprache als `expression` erhalten.

## Änderungen

- Cognitive Reasoning Prompt verschärft:
  - Fragen nach eigenen gespeicherten Materialien, Projektstand, TODOs, Test-Prompts, Vault/Memory/Dokumenten sollen `use_knowledge` oder `use_memory` wählen.
  - `answer_directly` ist dafür nicht zulässig.
- Tool-Contract-Guard ergänzt:
  - `calculator` wird nur ausgeführt, wenn tatsächlich eine arithmetische Expression vorhanden ist.
  - Bei freier Sprache wird auf `clarify` umgeleitet, statt Planner/Worker mit ungültiger Expression zu starten.
- Clarification-Antwort verbessert:
  - Primzahlen-/Bereichsfall fragt jetzt sauber: Python-Skript/einmalig lösen oder dauerhaftes Tool-Proposal anlegen?

## Tests

- `19 passed`
