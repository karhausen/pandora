# MVP 22.6.1 – LLM Routing Editor

Der LLM Routing Editor erweitert das LLM & Profile Center um kontrolliert editierbare Routing-Regeln.

## Ziel

Pandora soll transparent steuern können, welche Aufgabentypen an lokale Modelle, private Cloud-LLMs oder Company-LLMs gehen.

## GUI

Seite: `/llm-profiles`

Der Routing Editor erlaubt:

- Provider je Zweck ändern
- optional Modell überschreiben
- Änderungsgrund erfassen
- Preview vor dem Speichern
- Speichern als Local Override
- Audit anzeigen
- letztes Backup zurückspielen

## API

```text
GET  /api/gui/llm-profiles/routing-editor/status
GET  /api/gui/llm-profiles/routing-editor/routes
POST /api/gui/llm-profiles/routing-editor/preview
POST /api/gui/llm-profiles/routing-editor/apply
GET  /api/gui/llm-profiles/routing-editor/audit
POST /api/gui/llm-profiles/routing-editor/rollback
```

## Sicherheitsregeln

- Keine API-Keys in der GUI.
- Keine Base-URLs oder Secrets über den Editor speichern.
- Es werden nur `model_routes` in `config/llm/llm_config.local.json` geschrieben.
- Jede Änderung erzeugt einen Audit-Eintrag.
- Vor Änderung wird ein Backup der lokalen Config erzeugt, falls vorhanden.
- Company- und Cloud-Routing erzeugen Warnungen.

## Empfehlung

Routingänderungen zuerst per Preview prüfen. Cloud- und Company-Routing nur bewusst aktivieren.
