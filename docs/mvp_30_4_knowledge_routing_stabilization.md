# MVP 30.4 – Knowledge Routing Stabilization

Ziel dieses MVP ist bewusst klein:

```text
Userfrage
  ↓
Braucht die Antwort gespeichertes Wissen?
  ├─ Ja  → Vault/Knowledge sicher laden → LLM antwortet mit Kontext
  └─ Nein → direkt LLM antwortet ohne Vault-Kontext
```

Nicht Bestandteil von MVP 30.4:

- Tool-Ausführung
- Tool-Entwicklung
- Capability Gap
- Planner/Worker-Ausführung
- Evolution

## Neue Komponente

`core/knowledge_intent_router.py`

Diese Komponente entscheidet ausschließlich:

```python
needs_knowledge: bool
```

Sie darf keine Tools auswählen, keine Proposals erzeugen und keine Capability-Gaps behandeln.

## ChatService-Regel

`ChatService.run()` nutzt für MVP 30.4 nur zwei Pfade:

1. `answer_with_context` – bei `needs_knowledge=True`
2. `answer_directly` – bei `needs_knowledge=False`

## Regression-Ziele

Vault-Pfad:

- Welche Test-Prompts habe ich?
- Welche Todos habe ich?
- Was steht im Pandora-Projekt?

Direkter LLM-Pfad:

- Was ist Python?
- Was ist eine Primzahl?
- Erkläre FFT.

