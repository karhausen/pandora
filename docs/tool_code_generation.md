# Cloud Tool Code Generator

MVP 19.7 ergänzt den echten Code-Generierungs-Schritt nach dem Tool Design.

## Workflow

```text
Capability Gap
↓
Tool Development Agent
↓
Tool Design Agent
↓
Cloud Tool Code Generator
↓
Tool Proposal Manager
↓
Static Review
↓
pytest
↓
Proposal
↓
manuelle Aktivierung
```

## Verantwortung

Der `CloudToolCodeGenerator` erzeugt aus einem `ToolDesign`:

- Python-Modul mit `TOOL_META`
- `run(payload: dict) -> dict`
- pytest-Datei
- Implementierungs-/Sicherheitsnotizen

Er aktiviert kein Tool. Die Aktivierung bleibt manuell.

## Sicherheitsregeln

Der Generator-Prompt verbietet:

- hardcodierte Secrets
- Shell/Subprocess
- `eval`/`exec`
- Dateizugriff
- direkte Aktivierung

Die lokale Validierung bleibt maßgeblich. Cloud-Code ist nur ein Vorschlag.

## CLI

```bash
python main.py tool-generate word_count --provider mock
python main.py tool-generate word_count
```

Ohne `--provider mock` wird über den Model Router die Route `tool_generation` genutzt. Im privaten Profil ist das der Cloud Expert, aktuell `openai` mit `gpt-4o`.

## Ergebnis

Der Proposal-Ordner enthält:

```text
tool_proposals/<proposal_id>/
├─ generated_tools/<tool_id>.py
├─ tests/test_<tool_id>.py
├─ tool_design.json
├─ proposal.json
└─ validation.json
```
