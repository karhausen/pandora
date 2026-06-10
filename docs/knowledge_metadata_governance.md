# MVP 22.9 – Knowledge Metadata & Governance

Pandora unterstützt ab diesem Stand optionale Markdown-Metadaten in der User Knowledge Base.

## Beispiel

```yaml
---
title: Pandora Tool Factory
tags:
  - pandora
  - tools
visibility: public
cloud_allowed: true
priority: high
owner: thomas
last_reviewed: 2026-06-09
---
```

## Regeln

- `visibility` muss zum Ordner passen.
- `private_local_only` darf niemals `cloud_allowed: true` setzen.
- `last_reviewed` sollte gepflegt werden.
- `priority` beeinflusst das Ranking der Knowledge-Suche.
- Tags werden in der Suche und in der GUI angezeigt.

## CLI

```bash
python main.py knowledge-governance-status
python main.py knowledge-governance-run
python main.py knowledge-metadata-audit
```

## API

```text
GET  /api/gui/knowledge/governance
GET  /api/gui/knowledge/governance/status
GET  /api/gui/knowledge/metadata
POST /api/gui/knowledge/metadata/validate
```

## Wichtig

Die Governance ist read-only. Pandora verändert keine User-Knowledge-Dateien automatisch.


## MVP 22.9.1 Hardening

Die Governance prüft jetzt echte Regelverletzungen: fehlende Metadaten, Sichtbarkeitskonflikte, private Cloud-Freigaben, mögliche Secrets in public, Review-Alter, schwache Tags, sehr kurze/leere Inhalte, große Kontextdateien und Duplicate-Hinweise. Reports enthalten `health_score`, `grade` und eine klare Summary.
