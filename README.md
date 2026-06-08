# Pandora Agent

## MVP 21.1 – Nightly Governance Review

Ziel:

- Pandora kann einen Nacht-Review erzeugen, ohne sich selbst zu verändern.
- Der Review prüft Core-Status, Governance und Task-Historie.
- Ergebnis ist ein prüfbares Paket unter `proposals/nightly_reviews/`.
- `auto_changes_made` bleibt bewusst immer `false`.

CLI:

```bash
python main.py control-status
python main.py nightly-reflect --limit 50
python main.py nightly-review --limit 50
python main.py nightly-review --no-write
pytest tests/test_mvp21_control_core.py tests/test_mvp21_1_governance_review.py -q
```


## MVP 20.5 – Design Driven Code Generation & Placeholder Detection

Ziel:

- Tool-Code wird strikt aus dem `ToolDesign` erzeugt.
- `run(payload)` muss alle Felder aus `output_schema` liefern.
- Generischer Dummy-Code wie `return {"text": str(text)}` wird nicht mehr als brauchbares Tool akzeptiert, wenn das Schema andere Felder verlangt.
- Cloud-Fehler erzeugen keinen scheinbar gültigen Fallback-Code mehr. Stattdessen wird ein FAILED-Proposal mit transparenter Fehlermeldung erzeugt.
- `generate_with_llm` bewertet jetzt Static Review, pytest **und** Semantic Quality Gate gemeinsam.

Aktuelle CLI-Smoke-Tests:

```bash
python main.py status
python main.py llm-provider-smoke --live
python main.py tool-generate word_count --provider cloud_expert
python main.py proposal-list
python main.py tool-quality-proposal <proposal_id>
python main.py proposal-approve <proposal_id>
python main.py proposal-install <proposal_id>
python main.py tool-list
python main.py run-tool <tool_id> --file payload.json
```

Entwicklertests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q .
```



Lokaler, modularer Multi-Agent-Assistent mit kontrollierter Tool Factory.

Aktueller Stand: **MVP 20.4.1 – Implicit Live Data Gap Detection + No Dummy Code Policy**

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

## Aktuelle CLI-Tests für MVP 20.4.1

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
python3 main.py capability-evaluate "Ich brauche ein Tool um Aktienkurse abzurufen"
python3 main.py capability-evaluate "Wie ist der aktuelle Dollar-Kurs?"
python3 main.py capability-evaluate "Wie wird das Wetter?"
python3 main.py proposal-list
python3 main.py proposal-show <PROPOSAL_ID>
python3 main.py tool-quality-proposal <PROPOSAL_ID>
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





## MVP 20.4.1 Hinweis

MVP 20.4.1 ergänzt die implizite Live-Daten-Erkennung und eine No-Dummy-Code-Policy.

Beispiele:

```text
Wie wird das Wetter?
→ weather_lookup

Wie ist der Dollar-Kurs?
→ exchange_rate_lookup

Ich brauche ein Tool um Aktienkurse abzurufen
→ stock_price_lookup
```

Zusätzlich markiert das Tool Quality Gate generischen Dummy-Code wie `return {"text": str(text)}` als Fehler, wenn das `output_schema` andere Felder erwartet. Dadurch bleiben Proposals wie `stock_price_lookup` korrekt `FAILED`, solange der erzeugte Code den Design-Vertrag nicht erfüllt.

## MVP 20.4 Hinweis

MVP 20.4 verbessert die Capability-Erkennung vor dem Planner. Pandora erkennt jetzt generischer, wenn eine Aufgabe aktuelle/live Daten oder einen expliziten Tool-Wunsch enthält und kein installiertes Tool vorhanden ist.

Beispiele:

```text
Ich brauche ein Tool um Aktienkurse abzurufen
→ stock_price_lookup

Wie ist der aktuelle Dollar-Kurs?
→ exchange_rate_lookup

Welche Primzahlen liegen zwischen 10 und 30?
→ keine Capability-Lücke, normale lokale Antwort ist okay
```

Damit soll der Planner nicht mehr fälschlich mit `Keine Tool-Ausführung nötig` antworten, wenn tatsächlich ein neues Abruf-/Live-Daten-Tool benötigt wird.

## MVP 20.3 Hinweis

MVP 20.3 ergänzt ein Tool Quality Gate. Ein Proposal wird nur noch `VALIDATED`, wenn alle drei Ebenen erfolgreich sind:

```text
Static Review
pytest
Semantic Validation
```

Die semantische Validierung prüft den Vertrag aus `ToolDesign`/`output_schema` gegen das tatsächliche Tool-Ergebnis. Beispiel: Ein Tool mit `output_schema: {"count": "integer"}` darf nicht `{ "text": ... }` zurückgeben.

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
docs/tool_quality_gate.md
docs/security.md
```

## Release bereinigen

```bash
python scripts/clean_runtime_artifacts.py
```

## MVP 21.0 - Stable Control Core

Dieser Stand ergänzt Pandora um einen geschützten Control-Core-Pfad:

- zentrale Statusquelle: `core/core_status.py`
- Schaltzentrale: `core/control_core.py`
- Safety Gate: `core/safety_gate.py`
- Memory Gateway: `core/memory_gateway.py`
- Nachtreflexion ohne Auto-Aktivierung: `core/nightly_reflection.py`
- erweiterter Heartbeat für Planner, Memory, Registry und Tool Executor
- Dockerfile und docker-compose für reproduzierbaren API-Start

Wichtig: Der Core bleibt geschützt. Wachstum findet über Tools, Skills, Workflows und Memory statt.

## MVP 21.3 – Maintenance Manager

Pandora besitzt jetzt einen kontrollierten Wartungsmodus als Vorstufe zum späteren Day/Night-Mode.

```bash
python main.py maintenance-status
python main.py maintenance-run --dry-run --force
python main.py maintenance-run --force --limit 200
```

Der Maintenance Manager erzeugt Reviews und Reports, führt Audits aus und bleibt observe-only bezüglich Core, Tools und Skills. Er aktiviert keine Änderungen automatisch.


## MVP 21.3 – Skill Candidate Pipeline

Pandora kann im Wartungsmodus wiederkehrende Tool-Muster aus dem Task Journal erkennen und daraus prüfbare Skill-Vorschläge erzeugen. Die Pipeline ist observe-only: keine Aktivierung, keine Registry-Änderung, keine Core-Änderung.

```bash
python main.py skill-candidate-status
python main.py skill-candidate-run --dry-run --force
python main.py skill-candidate-run --force --limit 200
```

## MVP 21.5 – Capability Gap Pipeline

Pandora can now consolidate missing-capability signals from the capability event log and task journal into reviewable proposals.

Commands:

```bash
python main.py capability-gap-status
python main.py capability-gap-run --dry-run --force
python main.py capability-gap-run --force --limit 200
```

The pipeline is observe-only. It does not generate code, install tools, activate skills, call LLMs or modify the core.
