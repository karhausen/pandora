# MVP 25.8 – Knowledge Recommendation Workflow

## Ziel

Der Knowledge Recommendation Workflow wandelt erkannte Wissenslücken in reviewbare Verbesserungsvorschläge um.

Er ist bewusst keine Schreib- oder Veröffentlichungslogik.

## Position in der Cognitive Architecture

```text
Request Interpreter
        ↓
Capability Analyzer
        ↓
Python Orchestrator
        ↓
Knowledge Recommendation Workflow
        ↓
Review / Governance / User Approval
        ↓
Knowledge Base oder Obsidian Persistence
```

## Grundregel

```text
LLM erkennt mögliche Wissenslücken.
Python validiert den Plan.
Pandora erzeugt nur einen Vorschlag.
Der Benutzer gibt frei.
Erst danach darf Wissen persistiert werden.
```

## Sicherheitsgarantie

Der Workflow:

- schreibt nicht in den Obsidian Vault,
- schreibt nicht in die User Knowledge Base,
- verändert kein Memory,
- veröffentlicht keine Inhalte,
- erzeugt nur reviewbare Briefs.

## CLI

```bash
python main.py knowledge-recommendation-status
python main.py knowledge-recommendation-preview "Die Dokumentation fehlt für den Cognitive Layer"
```

## API

```text
GET /api/cognitive/knowledge-recommendation/status
GET /api/cognitive/knowledge-recommendation/preview?query=...
```

## Output

Der Preview-Output enthält:

- `knowledge_gap_count`
- `knowledge_improvement_briefs`
- `target_area`
- `recommended_artifact`
- `source_requirements`
- `proposal_contract`
- `review_workflow`
- `quality_checks`
- `safety`

## Review Workflow

```text
knowledge_gap_detected
        ↓
source_trace_review
        ↓
draft_review
        ↓
governance_check
        ↓
user_approval
        ↓
knowledge_or_obsidian_persistence
        ↓
post_update_learning_review
```

## Ergebnis

MVP 25.8 verbindet Capability Gap Detection mit kontrollierter Knowledge Evolution.

Pandora kann damit Wissenslücken erkennen und Verbesserungsvorschläge vorbereiten, ohne autonom Wissen zu verändern.
