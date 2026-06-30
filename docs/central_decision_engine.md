# MVP 26.1 – Central Decision Engine

Die Central Decision Engine ist Pandoras zentrale Freigabe- und Entscheidungsstelle im Cognitive Layer.

Sie sammelt die Ergebnisse aus:

- Request Interpreter
- Capability Analyzer
- Python Orchestrator
- Tool Recommendation Workflow
- Knowledge Recommendation Workflow
- Core Recommendation Workflow
- Working Memory

und erzeugt daraus **ein einziges Decision Object**.

## Grundregel

Das LLM empfiehlt. Python validiert. Die Central Decision Engine entscheidet den nächsten kontrollierten Schritt.

Sie führt nichts aus:

- keine Tool-Ausführung
- keine Code-Generierung
- keine Vault-/Knowledge-Schreibzugriffe
- keine Registry-Aktivierung
- keine Core-Änderung

## User Experience

Bei Tool-Gaps soll Pandora einfach fragen:

> Wir brauchen ein Tool `xy`. Soll ich den Tool-Vorschlag ausarbeiten?

Bei Core-Gaps:

> Ich sehe eine mögliche Core-Verbesserung `xy`. Soll ich einen prüfbaren Vorschlag ausarbeiten?

Erst nach Zustimmung wird der jeweilige Review-/Proposal-Prozess fortgesetzt.

## CLI

```bash
python main.py central-decision-status
python main.py central-decide "Baue ein Tool für historische Aktienkurse"
```

## API

```text
GET /api/cognitive/central-decision/status
GET /api/cognitive/central-decision/preview?query=...
```

## Decision Object

Wichtige Felder:

- `decision_type`
- `execution_mode`
- `requires_user_approval`
- `approval_prompt`
- `next_controlled_step`
- `gap_types`
- `review_packages`
- `working_memory`
- `safety`

## Architekturprinzip

Python verwaltet.
LLM versteht.
Python validiert.
LLM empfiehlt.
Python entscheidet.
Python handelt nur nach Freigabe.
