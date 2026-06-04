# Pandora Agent

Lokaler, modularer Multi-Agent-Assistent mit kontrollierter Tool-Entwicklung.

Aktueller Stand: **MVP 19.8.1 – Policy-Aware Test Generation**

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


## Aktuelle CLI-Tests für MVP 19.8.1

Diese Tests prüfen den aktuellen Stand ohne zusätzliche Tool-API-Keys wie `WEATHER_API_KEY`. Für Cloud-Tests muss nur `OPENAI_API_KEY` gesetzt sein.

```bash
python3 main.py status
python3 main.py config-paths
python3 main.py llm-profile-status
python3 main.py llm-provider-smoke cloud_expert --live
python3 main.py model-route tool_design
python3 main.py model-route tool_generation
python3 main.py tool-design word_count --provider cloud_expert
python3 main.py tool-generate word_count --provider cloud_expert
python3 main.py tool-proposal-list
python3 -m pytest -q
python3 -m compileall -q .
```

Optionaler Netzwerktool-Test, nur wenn ein Proposal für ein Netzwerktool geprüft werden soll:

```bash
python3 main.py tool-generate weather_lookup --provider cloud_expert
```

Erwartung bei `weather_lookup`: Tests müssen offline laufen, Netzwerkaufrufe müssen gemockt sein, benötigte ENV-Werte müssen im Test über `monkeypatch.setenv(...)` gesetzt werden.

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
python main.py tool-review-file tool_proposals/<PROPOSAL_ID>/generated_tools/<TOOL>.py --design tool_proposals/<PROPOSAL_ID>/tool_design.json
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




## MVP 19.8.1 – Policy-Aware Test Generation

Neu/Fix:

- Cloud-generierte pytest-Dateien werden nachbearbeitet, damit sie policy-konform und offline lauffähig sind.
- Fehlende Test-Imports wie `json` oder `urllib.request` werden ergänzt.
- Wenn generierter Tool-Code ENV-Variablen wie `WEATHER_API_KEY` benötigt, setzt der Test diese mit `monkeypatch.setenv(...)`.
- Verbotene Design-Dependencies wie `requests`, `httpx` oder `aiohttp` werden aus dem ToolDesign entfernt und in `risk_notes` dokumentiert.
- README enthält ab jetzt die aktuellen CLI-Tests für den jeweiligen Stand.
- Empfohlener Standardtest ohne Tool-API-Key: `tool-generate word_count --provider cloud_expert`.

Prüfung:

```bash
python3 -m pytest -q
python3 -m compileall -q .
```

## MVP 19.8 – Tool Review & Policy-Aware Validation

Neu:

- `ToolReviewAgent`
- policy-aware `ToolValidator`
- `LIMITED` Tools mit `requires_network=true` dürfen kontrolliert `urllib.request`, `urllib.parse`, `urllib.error`, `json` und `os` verwenden
- `requests`, `httpx`, `socket`, `subprocess`, `eval`, `exec`, `open` bleiben verboten
- Netzwerkaufrufe müssen `timeout=` setzen
- statische Reviews enthalten jetzt `policy`-Details
- CLI: `tool-review-file`
- API: `POST /tool-review/review`

Wichtig:

```text
Cloud erzeugt Code
↓
lokaler ToolReviewAgent prüft Security/Policy
↓
pytest läuft nur, wenn Review ok ist
↓
Proposal wird VALIDATED oder FAILED
```

Damit kann ein Wettertool mit Netzwerkbedarf kontrolliert validiert werden, ohne die generellen Sicherheitsregeln für SAFE-Tools aufzuweichen.

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
docs/tool_review.md
docs/security.md
```

## Release bereinigen

Vor dem Verpacken:

```powershell
python scripts/clean_runtime_artifacts.py
```
