# MVP 22.9.1 – Knowledge Governance Hardening

Dieser Patch macht die Knowledge Governance praktisch nutzbar. Die Prüfung ist weiterhin read-only: Pandora verschiebt, löscht oder ändert keine User-Dateien.

## Neue Prüfungen

- fehlender YAML-Header
- fehlende Pflichtfelder: `title`, `tags`, `visibility`, `cloud_allowed`, `priority`, `last_reviewed`
- ungültige `visibility`
- Abweichung zwischen Ordner und `visibility`
- `private_local_only` mit `cloud_allowed: true`
- Cloud-freigegebene Dateien mit vertraulichen Schlüsselwörtern
- public-Dateien mit möglichen Secret-Hinweisen
- veraltetes oder fehlendes `last_reviewed`
- fehlende oder schwache Tags
- sehr kurze oder leere Inhalte
- sehr große Kontextdateien
- doppelte Titel und doppelte Inhalte

## Health Score

Governance Reports enthalten jetzt:

- `health_score` von 0 bis 100
- `grade` A bis E
- Fehler-, Warnungs- und Hinweiszähler
- klare Summary

## CLI

```bash
python main.py knowledge-governance-run
python main.py knowledge-governance-status
python main.py knowledge-metadata-audit
```

## Beispiel für gute Markdown-Metadaten

```yaml
---
title: Pandora Tool Factory
tags:
  - pandora
  - tool-factory
visibility: public
cloud_allowed: true
priority: high
owner: thomas
last_reviewed: 2026-06-10
summary: Überblick über Tool-Erzeugung, Review und Installation.
---
```

## Harte Regel

`private_local_only` darf niemals mit `cloud_allowed: true` kombiniert werden.
