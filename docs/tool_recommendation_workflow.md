# MVP 25.7 – Tool Recommendation Workflow

## Ziel

Der Tool Recommendation Workflow verbindet erkannte Tool-Gaps mit der bestehenden Tool Factory.

Er baut **kein Tool direkt** und aktiviert nichts automatisch. Er erzeugt nur einen prüfbaren Tool-Factory-Brief mit:

- Zweck des fehlenden Tools
- stabiler Tool-Schnittstelle
- Input-/Output-Schema
- Sicherheitsregeln
- Testanforderungen
- Review- und Freigabeweg

## Grundregel

```text
LLM empfiehlt.
Python validiert.
Tool Factory erzeugt Vorschläge.
Tests und Governance prüfen.
User gibt frei.
Pandora nutzt.
```

## Pipeline

```text
User Request
  ↓
Request Interpreter
  ↓
Capability Analyzer
  ↓
Python Orchestrator
  ↓
Tool Recommendation Workflow
  ↓
Tool Factory Brief
  ↓
Review / Tests / Governance / User Approval
  ↓
Registry Activation
```

## Sicherheitsgrenzen

Der Workflow:

- generiert keinen Python-Code,
- schreibt keine Tool-Dateien,
- führt keine Tools aus,
- aktiviert keine Registry-Einträge,
- verändert keinen Core.

## CLI

```bash
python main.py tool-recommendation-status
python main.py tool-recommendation-preview "Baue ein Tool für historische Aktienkurse"
```

## API

```text
GET /api/cognitive/tool-recommendation/status
GET /api/cognitive/tool-recommendation/preview?query=Baue%20ein%20Tool
```
