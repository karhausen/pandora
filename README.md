# Pandora Agent

Lokaler, modularer Multi-Agent-Assistent mit kontrollierter Tool-Entwicklung.

Aktueller Stand: **MVP 19.7 – Cloud Tool Code Generator**

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py status
python main.py api
```

User-GUI:

```text
http://127.0.0.1:8000/
```

Admin-GUI:

```text
http://127.0.0.1:8000/admin
```

## Wichtige Befehle

Status und Tests:

```powershell
python main.py status
python main.py heartbeat
python -m pytest
python -m compileall .
```

Konfiguration:

```powershell
python main.py config-paths
python main.py llm-config-security
python main.py llm-profile-status
python main.py llm-profile private
python main.py llm-profile company
python main.py llm-provider-status cloud_expert
python main.py llm-provider-smoke cloud_expert --live
```

Model Routing:

```powershell
python main.py model-routes
python main.py model-route chat
python main.py model-route tool_selection
python main.py model-route tool_design
python main.py model-route tool_generation
```

Tools:

```powershell
python main.py tools
python main.py tool-design weather_lookup --task "Ich möchte das aktuelle Wetter abrufen" --provider mock
python main.py tool-propose-capability word_count
python main.py tool-generate word_count --provider mock
python main.py tool-generate word_count
python main.py tool-proposal-list
python main.py tool-proposal-show <PROPOSAL_ID>
python main.py tool-proposal-activate <PROPOSAL_ID>
```

Chat / Agent:

```powershell
python main.py planner-worker-run "Bitte rechne 2+3*4"
python main.py agent-run "Hallo Pandora" --provider mock
```

## Konfiguration

Statische Konfiguration liegt unter `config/`:

```text
config/
├─ llm/
│  ├─ llm_config.template.json
│  ├─ llm_config.json
│  └─ llm_config.local.json   # privat, gitignored
├─ tools/
│  ├─ tool_registry.json
│  └─ execution_policy.json
├─ skills/
│  └─ skill_registry.json
└─ system/
   └─ pandora.json
```

Runtime-Daten liegen unter `memory/` und gehören nicht in Git.

Private Secrets gehören in `.env` oder in lokale ENV-Variablen:

```env
OPENAI_API_KEY=...
COMPANY_LLM_BASE_URL=...
COMPANY_LLM_API_KEY=...
COMPANY_LLM_MODEL=...
```

`config/llm/llm_config.local.json` schaltet das Profil um:

```json
{
  "active_profile": "private"
}
```

oder:

```json
{
  "active_profile": "company"
}
```


## MVP 19.7 – Cloud Tool Code Generator

Neu:

- `CloudToolCodeGenerator`
- Tool-Code und pytest-Dateien werden aus dem `ToolDesign` erzeugt
- Route `tool_generation` nutzt den aktiven Cloud Expert
- OpenAI-Standardmodell ist jetzt `gpt-4o`
- Cloud-Code bleibt Proposal-Code und wird lokal geprüft
- Aktivierung bleibt manuell

Ablauf:

```text
Capability Gap
↓
Tool Design
↓
Cloud Tool Code Generator
↓
Static Review + pytest
↓
Proposal
↓
manuelle Aktivierung
```

Wichtig: Cloud erzeugt nur Kandidaten. Pandora validiert lokal und aktiviert nichts automatisch.

## MVP 19.6 – Real Tool Design Agent

Neu:

- `ToolDesignAgent`
- `ToolDesign` / `ToolDesignResult`
- `tool_design` Model-Route
- CLI: `tool-design`
- API: `POST /tool-design/design`
- Tool-Proposals enthalten jetzt `tool_design.json`
- Tool-Erzeugung startet mit einem Tool-Vertrag: Input, Output, Security, Netzwerkbedarf, Testfälle, Risiken

Ablauf:

```text
Capability Gap
↓
Tool Development Agent
↓
Tool Design Agent
↓
Tool Proposal Manager
↓
Validator / Tests
↓
Proposal
↓
manuelle Aktivierung
```

Wichtig: Auch MVP 19.6 aktiviert neue Tools nicht automatisch.

## Sicherheit

Pandora darf Tool-Vorschläge erzeugen und validieren, aber neuen Code nicht ungeprüft aktivieren. Aktivierung bleibt manuell.

Keine Zugangsdaten, Company-URLs oder privaten Profile ins Repository committen.

## Dokumentation

Weitere Details:

```text
docs/architecture.md
docs/roadmap.md
docs/tool_design.md
docs/tool_code_generation.md
docs/security.md
```

## Release bereinigen

Vor dem Verpacken:

```powershell
python scripts/clean_runtime_artifacts.py
```
