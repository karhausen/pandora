# MVP 25.2 – Context Builder Completion

## Ziel

MVP 25.2 erweitert den Cognitive Context Builder, ohne den bestehenden GUI-Chat-, Obsidian- oder Prompt-Flow aus MVP 25.1.3 aufzubrechen.

Der Context Builder bleibt der kontrollierte Python-Teil der Cognitive Pipeline: Er sammelt erlaubte Quellen, bereitet Kontext vor und liefert Diagnosedaten. Das LLM liest weiterhin keine Dateien selbst.

## Umsetzung

Neu eingeführt:

- `core/context_ranker.py`
- deterministisches Context Ranking
- Duplicate Removal
- Budget-/Packing-Logik
- Context Diagnostics
- Regressionstests für Obsidian Topics und letzte Notiz
- JSON-sichere Obsidian-Index-Erzeugung bei YAML-Datumswerten

## Pipeline

```text
User Request
  ↓
CognitiveContextBuilder
  ↓
KnowledgeContextService
  ↓
ContextCandidate Collection
  ↓
ContextRanker
  ↓
Duplicate Removal
  ↓
Budget / Context Packing
  ↓
Prompt Context
  ↓
LLM
```

## Wichtige Regel

Das LLM entscheidet nicht über Dateizugriff. Pandora/Python sammelt und filtert Kontext nach Governance-Regeln. Das LLM erhält ausschließlich vorbereiteten Kontext.

## Diagnostics

Die Context-Payload enthält jetzt zusätzlich:

- `context_ranking.candidate_count`
- `context_ranking.ranked_count`
- `context_ranking.unique_count`
- `context_ranking.selected_count`
- `context_ranking.duplicates_removed`
- `context_ranking.budget`
- je Quelle `context_rank`, `context_score`, `score_breakdown`

## Regression-Schutz

Explizit abgesichert:

- GUI-Chat-Vault-Topics bleiben direkte Pandora-Context-Antworten.
- Obsidian-Kontext bleibt im Chat verfügbar.
- „Was war meine letzte Notiz?“ nutzt den Obsidian-Kontext.
- Obsidian-Reindex stürzt bei YAML-Datumswerten nicht ab.
