# MVP 29.4.5 – Generic Tool Generator Architecture Fix

## Ziel

Der Tool Generator enthält kein Domänenwissen mehr. Fähigkeiten wie Wortzählung, Primzahlen, Wetter oder Börsendaten dürfen nicht über Python-Sonderfälle erkannt oder implementiert werden.

## Architekturregel

```text
User Task
↓
LLM Capability Gap Analyzer
↓
LLM Tool Design
↓
LLM Tool Code Generation
↓
Python Validation
↓
Proposal / Review / Approval
```

Python verwaltet, validiert und führt aus. Das LLM versteht und entwirft.

## Änderungen

- `core/tool_generator.py` vollständig generisch neu aufgebaut.
- `core/tool_test_generator.py` vollständig generisch neu aufgebaut.
- `_looks_like_prime_tool()` entfernt.
- `_looks_like_word_counter()` entfernt.
- Capability-spezifische Codezweige im lokalen Generator entfernt.
- Deterministische Fallbacks erzeugen nur noch schema-sichere Review-Scaffolds.
- `ToolProposalManager.propose_for_capability()` nutzt bei vorhandenem `ToolDesign` die designgetriebene Codegenerierung.
- Fallback-Designs werden nicht mehr als fachlich validierte Tools verkauft.
- `ToolDevelopmentAgent` übergibt den Originaltask an die Proposal-Erzeugung.

## Wichtige Konsequenz

Wenn kein echtes LLM für Tool Design/Code Generation erreichbar ist, darf Pandora kein echtes Fachtool vortäuschen. Dann entsteht höchstens ein reviewpflichtiger generischer Entwurf oder ein fehlgeschlagener Proposal-Stand. Das ist Absicht.

## Testfälle

- `python main.py selftest cli`
- `python main.py selftest api`
- `python main.py selftest integration`
- Prüfen, dass Generator-Dateien keine `_looks_like_*`-Methoden mehr enthalten.

