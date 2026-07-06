# MVP 29.5 – Core Evolution

Controlled Core Evolution for Pandora.

## Scope

- Core Health analysis
- Risk hotspot detection
- Refactoring candidates
- Review-only Core EvolutionProposals
- Proposal Queue integration
- CLI/API/GUI integration

## Safety rule

Core Evolution never edits, replaces, activates or deletes core files automatically.
All changes must go through Proposal Queue, review, tests and explicit user approval.

## CLI

```powershell
python main.py core-evolution status
python main.py core-evolution health
python main.py core-evolution analysis
python main.py core-evolution refactoring
python main.py core-evolution proposals
python main.py core-evolution enqueue
```

## API

- `/api/core-evolution/status`
- `/api/core-evolution/health`
- `/api/core-evolution/analysis`
- `/api/core-evolution/refactoring`
- `/api/core-evolution/proposals`
- `/api/core-evolution/enqueue`

## GUI

Maintenance → Core Evolution
