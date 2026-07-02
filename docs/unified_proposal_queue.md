# MVP 28.9 – Unified Proposal Queue

## Ziel

MVP 28.9 führt eine zentrale Queue für alle `EvolutionProposal`-Typen ein. Tool-, Skill-, Knowledge-, Workflow-, Core-, GUI-, Prompt-, Memory-, Personality- und Learning-Vorschläge werden nicht mehr verstreut betrachtet, sondern in einer gemeinsamen Review- und Priorisierungsansicht gesammelt.

## Sicherheitsprinzip

Die Unified Proposal Queue aktiviert keine Änderungen. Sie schreibt keinen generierten Code, verändert keine Core-Dateien und umgeht keine Tests. Sie sammelt, filtert, priorisiert und dokumentiert Entscheidungen. Jede Aktivierung bleibt an Review, Tests und Benutzerfreigabe gebunden.

## Neue Komponenten

- `core/proposal_queue/queue_schema.py`
- `core/proposal_queue/queue_storage.py`
- `core/proposal_queue/queue_manager.py`
- `web/proposal-queue.html`
- `web/proposal-queue.js`
- `web/proposal-queue.css`

## CLI

```bash
python main.py proposal-queue-status
python main.py proposal-queue-list --limit 100
python main.py proposal-queue-from-factory "Improve tool health reporting" --type tool
python main.py proposal-queue-import-prioritized --min-priority 60
python main.py proposal-queue-decide <queue_id> --decision reviewed --note "geprüft"
python main.py proposal-queue-history
python main.py proposal-queue-stats
```

## API

- `GET /api/proposal-queue/status`
- `GET /api/proposal-queue/items`
- `GET /api/proposal-queue/item/{item_id}`
- `POST /api/proposal-queue/enqueue`
- `POST /api/proposal-queue/from-factory`
- `POST /api/proposal-queue/import-prioritized`
- `POST /api/proposal-queue/item/{item_id}/decide`
- `GET /api/proposal-queue/history`
- `GET /api/proposal-queue/statistics`

## Datenfluss

```text
Observation
↓
Pattern Recognition
↓
Improvement Prioritization
↓
Evolution Factory
↓
Unified Proposal Queue
↓
Review
↓
Tests
↓
Benutzerfreigabe
↓
Aktivierung
```

## Ergebnis

Pandora besitzt ab diesem Release einen zentralen Eingang für alle zukünftigen Evolution-Proposals. Damit ist der Weg für MVP 29.0 – Proposal Generator vorbereitet.
