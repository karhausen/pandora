# Pandora Agent

Lokaler, modularer Multi-Agent-Assistent mit kontrollierter Tool-Entwicklung.

Aktueller Stand: **MVP 20.0 – Controlled Tool Factory**

## Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py status
python3 main.py api
```

User-GUI: `http://127.0.0.1:8000/`
Admin-GUI: `http://127.0.0.1:8000/admin`

## Aktuelle CLI-Tests für MVP 20.0

Standardtests ohne zusätzliche Tool-API-Keys:

```bash
python3 main.py status
python3 main.py config-paths
python3 main.py llm-profile-status
python3 main.py llm-provider-smoke cloud_expert --live
python3 main.py model-route tool_design
python3 main.py model-route tool_generation
python3 main.py tool-generate word_count --provider cloud_expert
python3 main.py proposal-list
python3 main.py proposal-show <PROPOSAL_ID>
python3 main.py proposal-approve <PROPOSAL_ID>
python3 main.py proposal-install <PROPOSAL_ID> --test-json '{"text":"eins zwei drei"}'
python3 main.py tool-list
python3 main.py run-tool word_count --json '{"text":"eins zwei drei"}'
python3 -m pytest -q
python3 -m compileall -q .
```

Optionaler Netzwerktool-Test:

```bash
python3 main.py tool-generate weather_lookup --provider cloud_expert
```

Netzwerktools bleiben `LIMITED`, müssen offline testbar sein und werden erst nach manueller Prüfung installiert.

## Wichtige Befehle

Konfiguration/Profile:

```bash
python3 main.py config-paths
python3 main.py llm-config-security
python3 main.py llm-profile-status
python3 main.py llm-profile private
python3 main.py llm-profile company
python3 main.py llm-provider-status cloud_expert
python3 main.py llm-provider-smoke cloud_expert --live
```

Tool Factory:

```bash
python3 main.py tool-generate word_count --provider cloud_expert
python3 main.py proposal-list
python3 main.py proposal-show <PROPOSAL_ID>
python3 main.py proposal-approve <PROPOSAL_ID>
python3 main.py proposal-reject <PROPOSAL_ID>
python3 main.py proposal-install <PROPOSAL_ID> --test-json '{"text":"eins zwei drei"}'
python3 main.py tool-list
python3 main.py run-tool word_count --json '{"text":"eins zwei drei"}'
```

Tests/Status:

```bash
python3 main.py heartbeat
python3 -m pytest -q
python3 -m compileall -q .
```

## Konfiguration

Statische Konfiguration liegt unter `config/`, Runtime-Daten unter `memory/`.

```text
config/llm/llm_config.json
config/llm/llm_config.local.json   # privat, gitignored
config/tools/tool_registry.json
config/tools/execution_policy.json
config/skills/skill_registry.json
```

Secrets gehören in `.env` oder echte Umgebungsvariablen:

```env
OPENAI_API_KEY=...
COMPANY_LLM_BASE_URL=...
COMPANY_LLM_API_KEY=...
COMPANY_LLM_MODEL=...
```

Profilumschaltung:

```bash
python3 main.py llm-profile private
python3 main.py llm-profile company
```



## MVP 20.0.1 – Tool Install Metadata Normalization Hotfix

Fix:

- `proposal-install` kann jetzt Cloud-generierte `TOOL_META`-Varianten installieren, die design-style Felder wie `tool_id` enthalten.
- Beim Installieren normalisiert Pandora `spec`, `design` und `TOOL_META` zu gültigem `ToolMeta`.
- Runtime-Feld `module` wird automatisch auf `generated_tools.<tool_id>` gesetzt.
- Regressionstest für `word_count_tool` mit fehlendem `id`/`module` ergänzt.

Aktuelle CLI-Tests:

```bash
python3 main.py status
python3 main.py tool-list
python3 main.py proposal-list
python3 main.py proposal-approve <proposal_id>
python3 main.py proposal-install <proposal_id> --test-json '{"text":"eins zwei drei"}'
python3 -m pytest -q
python3 -m compileall -q .
```

Prüfung für dieses Paket:

- `pytest`: 67 passed
- `compileall`: erfolgreich

## MVP 20.0 – Controlled Tool Factory

Neu:

- Proposal-Lifecycle: `VALIDATED → APPROVED → INSTALLED` oder `REJECTED`
- Installation nur nach explizitem Approval
- installierte Tools werden nach `generated_tools/` kopiert und in `config/tools/tool_registry.json` registriert
- neue CLI-Aliase: `proposal-list`, `proposal-show`, `proposal-approve`, `proposal-reject`, `proposal-install`, `tool-list`
- API-Endpunkte für Approve/Reject/Install
- Release-Clean entfernt Runtime-Proposals und generierte Testtools

Akzeptanztest:

```bash
python3 main.py tool-generate word_count --provider cloud_expert
python3 main.py proposal-approve <PROPOSAL_ID>
python3 main.py proposal-install <PROPOSAL_ID> --test-json '{"text":"eins zwei drei"}'
python3 main.py run-tool word_count --json '{"text":"eins zwei drei"}'
```

Prüfung:

```bash
python3 -m pytest -q
python3 -m compileall -q .
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
