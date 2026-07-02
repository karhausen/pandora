# Pandora Agent – MVP 28.7 Pattern Recognition Engine

Dieser Release baut auf MVP 28.6 auf und ergänzt eine Pattern Recognition Engine für die Self-Observation-Daten.

## Neu in MVP 28.7

- `core/pattern/` als eigenes Paket
- Pattern Schema für erkannte Muster
- Rule-based Pattern Detector
- Pattern Storage mit SQLite-Schicht
- Pattern Recognition Manager und Engine
- Detektoren für:
  - häufige Event-Typen
  - wiederkehrende Komponentenfehler
  - langsame Komponenten
  - wiederholte Capability Gaps
  - Review-Entscheidungsmuster
  - GUI-Nutzungsschwerpunkte
- CLI-Befehle für Pattern Recognition
- API-Endpunkte unter `/api/pattern/*`
- Maintenance-Link und neue Seite `/pattern`
- Genome-Konfiguration für Pattern Recognition

## Wichtige Grenze

MVP 28.7 erkennt Muster in vorhandenen Observation-Fakten.
Es erzeugt keine Proposals, aktiviert keine Änderungen und verändert keine Core-, Tool-, Skill- oder Genome-Dateien automatisch.

Der nächste logische Schritt ist MVP 28.8 – Improvement Prioritization.

## CLI Smoke Tests

```bash
python main.py pattern-status
python main.py pattern-health
python main.py pattern-statistics
python main.py pattern-detect --limit 500
python main.py pattern-detect --limit 500 --save
python main.py pattern-list --limit 20
```

## API

- `GET /api/pattern/status`
- `GET /api/pattern/health`
- `GET /api/pattern/detect`
- `GET /api/pattern/patterns`
- `GET /api/pattern/statistics`

## Release-Hinweis

Runtime-, Test- und Build-Artefakte sind nicht Teil der Clean-ZIP.


## MVP 28.9 – Unified Proposal Queue

Zentrale Review-Queue für alle EvolutionProposal-Typen. Die Queue sammelt, filtert, priorisiert und dokumentiert Entscheidungen, aktiviert aber keine Änderungen ohne Review, Tests und Benutzerfreigabe.


## MVP 28.9.1 – CLI/API Alias Fix

Dieses Fix-Release hält die vorhandenen flachen CLI-Kommandos weiter kompatibel und ergänzt die dokumentierte, besser lesbare 28.x-Schreibweise.

Beispiele:

```powershell
python main.py genome status
python main.py genome validate
python main.py evolution status
python main.py evolution-factory status
python main.py observation status
python main.py pattern status
python main.py priority status
python main.py proposal-queue status
```

Zusätzlich wurden API-Aliase ergänzt, z. B. `/api/genome/status`, `/api/evolution-factory/status`, `/api/pattern-recognition/status` und `/api/priority/status`.

## MVP 28.9.2 – CLI & API Integration Hardening

Dieser Release ergänzt eine harte Integrationsprüfung für die dokumentierten Evolution-Kommandos.

Wichtige Tests:

```bash
python main.py genome status
python main.py proposal-queue add --type TOOL --title "Test Tool Proposal" --priority MEDIUM
python main.py proposal-queue list
python main.py selftest cli
python main.py selftest api
python main.py selftest integration
```


## MVP 29.0 – Proposal Generator

- Neuer `core/proposal_generator`-Layer für kontrollierte Proposal-Entwürfe.
- Neue CLI: `proposal-generator status|prompt|generate|enqueue|batch`.
- Neue API: `/api/proposal-generator/*`.
- Neue GUI: `/proposal-generator` im Maintenance Center.
- Sicherheitsmodus: review-only, keine Aktivierung, keine Code-Ausführung, keine Core-Änderung ohne Benutzerfreigabe.

## MVP 29.1 – Proposal Evolution

MVP 29.1 ergänzt eine kontrollierte Proposal-Evolution:

- bestehende Evolution-Proposals versionieren
- Versionen vergleichen
- Diffs erzeugen
- Proposals kontrolliert verbessern
- verbesserte Entwürfe optional wieder in die Unified Proposal Queue legen

Wichtig: Proposal Evolution aktiviert keine Änderungen automatisch. Jede Änderung bleibt Proposal-/Review-only und benötigt weiterhin Tests sowie Benutzerfreigabe.

### CLI

```powershell
python main.py proposal-evolution status
python main.py proposal-evolution snapshot --json '{"id":"demo","type":"tool","title":"Demo","description":"Base"}'
python main.py proposal-evolution improve "Tests und Risiko klarer beschreiben" --json '{"id":"demo","type":"tool","title":"Demo","description":"Base"}'
python main.py proposal-evolution history --proposal-id demo
python main.py proposal-evolution compare demo --from-version 1 --to-version 2
```

### API

```text
GET  /api/proposal-evolution/status
POST /api/proposal-evolution/snapshot
POST /api/proposal-evolution/snapshot-queue
GET  /api/proposal-evolution/history
GET  /api/proposal-evolution/compare/{proposal_id}
POST /api/proposal-evolution/diff
POST /api/proposal-evolution/improve
```

### GUI

```text
Maintenance → Proposal Evolution
```

## MVP 29.2 – Adaptive Goals

MVP 29.2 ergänzt langfristige, kontrollierte Ziele für Pandora:

- strategische, taktische und operative Goals
- Goal Status, Liste, Detailansicht und Historie
- Goal Evaluation
- konservative Repriorisierung
- API-/CLI-/GUI-Anbindung
- Integration in die CLI/API-Selftests

Wichtig: Adaptive Goals erzeugt keine automatische Umsetzung. Ziele dienen als Entscheidungsgrundlage; konkrete Änderungen laufen weiter über Proposal, Review, Tests und Benutzerfreigabe.

### CLI

```powershell
python main.py goals status
python main.py goals list
python main.py goals evaluate
python main.py goals reprioritize
```

### API

```text
GET  /api/goals/status
GET  /api/goals/list
GET  /api/goals/evaluate
POST /api/goals/reprioritize
```
