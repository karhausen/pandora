# User Knowledge Base

Die User Knowledge Base ist der Bereich, in dem du eigenes Wissen im Dateisystem ablegen kannst.
Pandora kann dieses Wissen suchen und als Kontext nutzen.

## Struktur

```text
user_knowledge/
├── public/
├── restricted_cloud_allowed/
└── private_local_only/
```

## Policy

```text
public                    darf lokal und in Cloud-Kontext genutzt werden
restricted_cloud_allowed  darf nach Policy-Prüfung in Cloud-Kontext genutzt werden
private_local_only        darf niemals an Cloud-LLMs gehen
```

## Empfohlenes Dateiformat

Markdown mit YAML-Metadaten:

```markdown
---
title: Beispielwissen
tags:
  - pandora
  - beispiel
visibility: public
cloud_allowed: true
priority: medium
owner: thomas
last_reviewed: 2026-06-10
---

# Beispielwissen

Hier steht der eigentliche Inhalt.
```

## Governance-Regeln

Pandora prüft unter anderem:

- fehlende Metadaten
- ungültige Sichtbarkeit
- Widerspruch zwischen Ordner und YAML-Header
- `private_local_only` mit `cloud_allowed: true`
- fehlende Tags
- fehlendes oder altes `last_reviewed`
- mögliche Secrets in öffentlichen Dateien
- doppelte Inhalte oder Titel

## Bedienung

GUI:

```text
/knowledge-base
```

CLI:

```bash
python main.py knowledge-governance-run
python main.py knowledge-metadata-audit
```

## Empfehlung

Lege erst wenige, hochwertige Markdown-Dateien an. Lieber klare Themen und gute Metadaten als viele ungepflegte Notizen.
