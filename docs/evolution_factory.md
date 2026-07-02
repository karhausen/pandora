# MVP 28.5 – Evolution Factory

## Ziel

Die Evolution Factory ist der zentrale Einstiegspunkt für kontrollierte Weiterentwicklung in Pandora. Sie ersetzt noch nicht alle bestehenden Detail-Workflows, aber sie normalisiert neue Verbesserungsideen auf ein gemeinsames `EvolutionProposal`-Modell.

## Prinzip

Die Factory erzeugt ausschließlich Vorschläge.

Sie führt nicht aus, aktiviert nicht automatisch und schreibt keine Runtime-Dateien.

```text
Request / Legacy Proposal
↓
Type Detection
↓
Evolution Factory Route
↓
EvolutionProposal
↓
Review / Queue / Approval in späteren MVPs
```

## Unterstützte Proposal-Typen

- tool
- skill
- knowledge
- workflow
- core
- gui
- prompt
- memory
- personality
- learning

## CLI

```bash
python main.py evolution-factory-status
python main.py evolution-factory-routes
python main.py evolution-factory-preview "Tool für Seriennummern-Prüfung verbessern" --type tool
python main.py evolution-factory-create --json '{"type":"knowledge","title":"Obsidian Coverage prüfen","description":"Wissenslücken erkennen"}'
python main.py evolution-factory-migration-plan
```

## API

```text
GET  /api/evolution/factory/status
GET  /api/evolution/factory/routes
GET  /api/evolution/factory/preview?request=...&type=tool
POST /api/evolution/factory/proposals
POST /api/evolution/factory/batch-preview
GET  /api/evolution/factory/migration-plan
```

## Sicherheitsvertrag

Jedes Proposal enthält im Payload einen Safety Contract:

```json
{
  "proposal_only": true,
  "requires_review": true,
  "requires_user_approval": true,
  "may_write_runtime_files": false
}
```

Damit bleibt das Grundprinzip erhalten:

> Pandora schlägt vor. Thomas entscheidet. Python führt kontrolliert aus.
