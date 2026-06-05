# Pandora Agent

Lokaler, modularer Multi-Agent-Assistent mit kontrollierter Tool Factory.

Aktueller Stand: **MVP 20.2.2 – Generated Tool Output Contract Hotfix**

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

## Aktuelle CLI-Tests für MVP 20.2.2

Grundstatus:

```bash
python3 main.py status
python3 main.py config-paths
python3 main.py llm-profile-status
python3 main.py llm-provider-smoke cloud_expert --live
```

Tool Factory mit sicherem Standardtool:

```bash
python3 main.py tool-generate word_count --provider cloud_expert
python3 main.py proposal-list
python3 main.py proposal-show <PROPOSAL_ID>
python3 main.py proposal-approve <PROPOSAL_ID>
python3 main.py proposal-install <PROPOSAL_ID>
python3 main.py tool-list
```

Payload-Datei für Windows/macOS/Linux:

```json
{
  "text": "eins zwei drei vier"
}
```

Tool ausführen:

```bash
python3 main.py run-tool <TOOL_ID> --file payload.json
```

Wichtig: Wenn `tool-list` z.B. `word_counter` mit Alias `word_count` zeigt, funktionieren beide Ebenen im Routing: Pandora erkennt die Capability `word_count`, nutzt aber das installierte Tool `word_counter`.

Lifecycle:

```bash
python3 main.py tool-info <TOOL_ID>
python3 main.py tool-stats <TOOL_ID>
python3 main.py tool-disable <TOOL_ID>
python3 main.py tool-enable <TOOL_ID>
```

Tests:

```bash
python3 -m pytest -q
python3 -m compileall -q .
```


## MVP 20.2.2 Hinweis

Der Generator prüft jetzt strenger, dass erzeugter Tool-Code zum `output_schema` passt. Ein Tool mit `output_schema: {"count": "integer"}` darf nicht nur `{ "text": ... }` zurückgeben.

Der Clean-Release-Schritt entfernt außerdem installierte Generated Tools aus `generated_tools/` und setzt `config/tools/tool_registry.json` auf die Basis-Tools zurück.

## GUI Workflow

1. User fragt nach einer neuen Fähigkeit.
2. Pandora erzeugt ein Proposal.
3. GUI öffnet den Tool-Factory-Bereich.
4. User klickt `Approve`.
5. User klickt `Install`.
6. Tool ist in `tool-list` sichtbar und kann verwendet werden.

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

OpenAI-Standardmodell: `gpt-4o`.

## Sicherheit

Cloud-Modelle erzeugen nur Kandidaten. Pandora validiert lokal. Aktivierung bleibt manuell.

Keine Zugangsdaten, Company-URLs oder privaten Profile ins Repository committen.

## Dokumentation

```text
docs/architecture.md
docs/roadmap.md
docs/tool_factory.md
docs/tool_factory_gui.md
docs/tool_lifecycle.md
docs/tool_design.md
docs/tool_code_generation.md
docs/tool_review.md
docs/security.md
```

## Release bereinigen

```bash
python scripts/clean_runtime_artifacts.py
```
