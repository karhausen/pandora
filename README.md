# Pandora Agent – MVP 29.4 Tool Evolution

Dieser Release baut auf MVP 29.3 auf und ergänzt eine kontrollierte Tool-Evolution-Schicht.

## Neu in MVP 29.4

- `core/tool_evolution/` als eigenes Paket
- Tool Health Score je Tool
- Tool Reviews auf Basis von Status, Nutzung, Fehlern und Laufzeit
- Lifecycle-Übersicht für Active, Validated, Experimental, Deprecated, Disabled, Failed und Archived
- Refactoring-/Verbesserungskandidaten als review-only EvolutionProposals
- CLI-Befehle und verschachtelte Aliase
- API-Endpunkte unter `/api/tool-evolution/*`
- Maintenance-Seite `/tool-evolution`
- Erweiterte CLI/API/Integration-Selftests

## Wichtige Grenze

Tool Evolution analysiert und schlägt vor. Es verändert keine Tool-Dateien automatisch, aktiviert keine Tools automatisch und umgeht niemals Review, Tests oder Benutzerfreigabe.

## CLI Smoke Tests

```bash
python main.py tool-evolution status
python main.py tool-evolution health
python main.py tool-evolution reviews
python main.py tool-evolution lifecycle
python main.py tool-evolution proposals
python main.py selftest cli
python main.py selftest api
python main.py selftest integration
```

## API

- `GET /api/tool-evolution/status`
- `GET /api/tool-evolution/health`
- `GET /api/tool-evolution/reviews`
- `GET /api/tool-evolution/lifecycle`
- `GET /api/tool-evolution/proposals`
- `POST /api/tool-evolution/enqueue`
- `GET /api/tool-evolution/history`

## Release-Hinweis

Runtime-, Test- und Build-Artefakte sind nicht Teil der Clean-ZIP.


## MVP 29.5 – Core Evolution

Core Evolution analysiert Core Health, Risiko-Hotspots und Refactoring-Kandidaten. Es erzeugt nur reviewpflichtige Proposals und ändert niemals automatisch Core-Dateien.


## MVP 29.6 – Decision Learning

Pandora records user decisions on Evolution Proposals and derives advisory decision patterns.

Useful commands:

```bash
python main.py learning status
python main.py learning history
python main.py learning patterns
python main.py learning statistics
python main.py learning influence
```

Decision Learning never activates changes automatically.
