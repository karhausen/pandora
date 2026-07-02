# MVP 29.0 – Proposal Generator

Der Proposal Generator startet Phase 4 – Controlled Evolution.

## Ziel

Pandora kann aus einer Anfrage einen kontrollierten `EvolutionProposal`-Entwurf erzeugen.
Der Entwurf ist immer reviewpflichtig und wird niemals automatisch aktiviert.

## Sicherheitsprinzipien

- keine automatische Aktivierung
- kein Merge
- kein Schreiben von generiertem Code
- keine Core-Änderung ohne Review, Tests und Benutzerfreigabe
- LLM ist optional; ohne LLM wird ein deterministischer lokaler Entwurf erzeugt

## CLI

```bash
python main.py proposal-generator status
python main.py proposal-generator generate "Tool-Fehler besser erkennen" --type TOOL
python main.py proposal-generator enqueue "GUI Review vereinfachen" --type GUI
python main.py proposal-generator prompt "Knowledge-Lücke erkennen" --type KNOWLEDGE
python main.py proposal-generator batch --file batch.json --enqueue
```

## API

```text
GET  /api/proposal-generator/status
POST /api/proposal-generator/prompt
POST /api/proposal-generator/generate
POST /api/proposal-generator/enqueue
POST /api/proposal-generator/batch
```

## GUI

```text
/maintenance → Evolution → Proposal Generator
/proposal-generator
```

## Pipeline

```text
Request / Pattern / Prioritized Candidate
↓
Proposal Generator
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
