# Pandora Agent

Lokaler modularer KI-Agent mit stabilem Core, Tool-/Skill-Evolution, Learning Layer und Web-GUI.

## Projektziel

Pandora soll Aufgaben analysieren, Tools und Skills kontrolliert nutzen, aus Erfahrungen lernen und neue Fähigkeiten sicher vorschlagen.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py status
python main.py api
```

Web-GUI:

```text
http://127.0.0.1:8000
```

## CLI

Wichtige Befehle:

```powershell
python main.py status
python main.py heartbeat
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
python main.py tools
python main.py skills
python main.py learn-from-journal
python main.py recommendations
python main.py docs-generate
python main.py governance-check
```

## API

FastAPI stellt Status-, Agent-, Tool-, Skill-, Capability-, Learning- und Dokumentations-Endpunkte bereit.

## Sicherheit

Der aktive Core darf nicht unkontrolliert überschrieben werden. Kritische Core-Dateien sind geschützt. Neue Tools und Skills entstehen zuerst als Proposal und werden erst nach Validierung und expliziter Aktivierung übernommen.

## Architektur

Siehe `docs/architecture.md`.

## Dokumentation

Weitere Dokumentation befindet sich unter `docs/`.

## Roadmap

Siehe `docs/roadmap.md`.


## MVP 14.1 – Web GUI Fix

- Dashboard vollständig wiederhergestellt
- Agent Run, Heartbeat, Tools, Skills, Journal, Proposals, Learning und Governance sichtbar
- JavaScript nutzt korrekt `provider_name` für `/agent/run`


## MVP 15.0 – Sandbox & Isolation System

Neu:

- ExecutionPolicyManager
- PermissionManager
- ProcessGuard
- ResourceMonitor
- IsolationRunner
- Sandbox
- SandboxLog
- ToolExecutor nutzt standardmäßig Sandbox-Ausführung
- CLI/API für Sandbox-Policies, Sandbox-Logs und isolierte Tool-Ausführung

Beispiele:

```powershell
python main.py sandbox-run-tool calculator --json "{\"expression\":\"2+3*4\"}"
python main.py sandbox-policies
python main.py sandbox-logs
```

Hinweis: MVP 15 bietet Prozess-Isolation und Timeouts. Harte OS-Level CPU/RAM-Limits sind für eine spätere Stufe vorgesehen.


## MVP 16.0 – Real Autonomous Tool Generation

Neu:

- LLMToolGenerator
- ToolCodePrompt
- ToolGenerationRunner
- ToolRepairManager
- ToolGenerationLog
- ToolProposalManager.generate_with_llm()
- CLI/API für LLM-gestützte Tool-Erzeugung

Beispiele:

```powershell
python main.py tool-generate word_count --provider mock
python main.py tool-generation-logs
python main.py tool-proposal-list
python main.py tool-proposal-activate <ID>
```

Sicherheitsregel: Auch MVP 16 aktiviert generierte Tools nicht automatisch. Es erzeugt validierte Proposals.


## MVP 16.1 – Tool Generation Stabilisierung

Neu:

- `tool-generate --no-tests` für schnelle lokale Smoke-Checks
- API-Parameter `run_tests`
- Web-GUI-Panel für Tool Generation
- Tool-Generation-Logs direkt sichtbar
- README/Dashboard nachgezogen

Beispiele:

```powershell
python main.py tool-generate text_reverse --provider mock --no-tests
python main.py tool-generation-logs
```

Für vollständige Validierung ohne `--no-tests`:

```powershell
python main.py tool-generate text_reverse --provider mock
```


## MVP 17.0 – Core Governance & Survival Layer

Neu:

- CoreVersionManager
- CoreSnapshot
- CoreSmokeRunner
- ActivationManager
- RollbackManager
- StabilityMonitor
- Core-Versionen unter `core_versions/`
- CLI/API für Snapshots, Smoke-Tests, Aktivierung und Rollback
- Dashboard-Kachel `Core Status`

Beispiele:

```powershell
python main.py core-status
python main.py core-smoke
python main.py core-snapshot --notes "stable after MVP 17"
python main.py core-versions
python main.py core-rollback
```

Hinweis: MVP 17 markiert Rollbacks und verwaltet Snapshots. Das automatische physische Ersetzen des aktiven Core bleibt bewusst noch manuell, damit der aktive Core nicht unkontrolliert überschrieben wird.


## MVP 17.1 – Reality Check

Neu:

- RealityCheck
- StabilityReporter
- RealityCheckLog
- Dauerlauf-artige Stabilitätsprüfung
- Snapshot-/Memory-Größenreport
- Empfehlungen nach Diagnose
- CLI/API/Dashboard für Reality Checks

Beispiele:

```powershell
python main.py reality-check --iterations 5 --delay 1
python main.py stability-report
python main.py reality-logs
```

Optional mit pytest pro Iteration:

```powershell
python main.py reality-check --iterations 1 --pytest
```


## MVP 18.0 – Planner Agent

Neu:

- PlannerAgent
- TaskPlan / PlanStep Modelle
- TaskPlanStore
- PlannerAgentLog
- CLI/API/Dashboard für strukturierte Planung

Beispiele:

```powershell
python main.py planner-plan "Bitte rechne 2+3*4" --provider mock
python main.py planner-plans
python main.py planner-logs
```

MVP 18 trennt Planung und Ausführung konzeptionell. Der PlannerAgent erzeugt zunächst nur strukturierte Pläne; die Worker-Ausführung folgt in MVP 18.1.


## MVP 18.1 – Worker Agent

Neu:

- WorkerAgent
- WorkerStepResult / TaskExecutionResult Modelle
- TaskExecutionStore
- WorkerAgentLog
- PlannerWorkerOrchestrator
- CLI/API/Dashboard für Plan-Ausführung

Beispiele:

```powershell
python main.py planner-plan "Bitte rechne 2+3*4" --provider mock
python main.py planner-plans
python main.py worker-execute-plan <PLAN_ID>
python main.py planner-worker-run "Bitte rechne 2+3*4" --provider mock
python main.py worker-executions
```

Hinweis: In der Build-Notebook-Umgebung kann der CLI-Smoke mit Sandbox-Subprozessen hängen. Die Unit-/API-Tests prüfen die Worker-Funktionalität erfolgreich.


## MVP 18.2 – User GUI

Neu:

- `/` ist jetzt die einfache User-GUI
- `/admin` ist das bisherige Admin-Dashboard
- `web/user.js`
- `web/user.css`
- `POST /user/run`
- `GET /user/status`
- kompakte Antwort für normale Nutzer
- Plan und Ausführung einklappbar sichtbar

Start:

```powershell
python main.py api
```

User-GUI:

```text
http://127.0.0.1:8000/
```

Admin-Dashboard:

```text
http://127.0.0.1:8000/admin
```


## MVP 18.3 – Chat Session Layer

Neu:

- ChatSessionStore
- ChatService
- ChatMessage / ChatSession / ChatRunResult Modelle
- `POST /chat/run`
- `POST /chat/sessions`
- `GET /chat/sessions`
- `GET /chat/sessions/{session_id}`
- `DELETE /chat/sessions/{session_id}`
- User-GUI mit Chat-Verlauf und Session-Auswahl

Die Startseite `/` bleibt die User-GUI, `/admin` bleibt das Admin-Dashboard.


## MVP 18.3.1 – User Response Fix

Fix:

- Begrüßungen wie `Hallo Pandora` liefern jetzt eine freundliche Antwort.
- Technischer Fallback `No suitable tool or skill needed.` wird in der User-GUI nicht mehr direkt angezeigt.
- Neue Komponente: `UserResponseFormatter`.


## MVP 18.3.2 – LLM Chat Response

Neu/Fix:

- Freie Texte und normale Chat-Nachrichten gehen jetzt an den LLM-Chat-Modus.
- Tool-nahe Aufgaben wie Berechnungen laufen weiter über PlannerAgent + WorkerAgent.
- Neue Komponenten:
  - `ChatResponseRouter`
  - `LLMChatResponder`
- User-GUI zeigt neueste Frage/Antwort oben.
- Chat-Verlauf bleibt in Sessions gespeichert.

Beispiele:

```text
Hallo Pandora
```

läuft über Chat-Modus.

```text
Bitte rechne 2+3*4
```

läuft über Planner/Worker/Tool.


## MVP 18.3.3 – Stale Chat Session Fix

Fix:

- Alte `session_id` im Browser-`localStorage` führt nicht mehr zu `500 Internal Server Error`.
- Wenn eine Session serverseitig fehlt, erzeugt `ChatService` automatisch eine neue Session.
- `GET /chat/sessions/{session_id}` liefert jetzt sauber `404`.
- User-GUI entfernt ungültige Session-IDs automatisch aus `localStorage`.


## MVP 18.3.4 – User-GUI Provider Auswahl

Neu:

- Provider-Auswahl in der User-GUI
- optionales Modellfeld
- Provider/Modell werden im Browser gespeichert
- `/chat/run` erhält `provider_name` und `model` aus der GUI
- `/user/status` liefert verfügbare Provider

Damit kann LM Studio direkt aus der User-GUI getestet werden, z.B. mit `local_fast` oder `lmstudio`.


## MVP 18.4 – Conversation Memory

Neu:

- ConversationMemory
- ConversationMemoryLog
- ConversationContext
- einfache Faktenextraktion aus Gesprächen
- Chat-Antworten erhalten Gesprächskontext
- Memory-Fragen wie `Wie heiße ich?` können aus gespeicherten Fakten beantwortet werden
- API:
  - `GET /memory/conversation`
  - `DELETE /memory/conversation/{key}`
  - `GET /memory/conversation/logs`
- CLI:
  - `python main.py conversation-memory`
  - `python main.py conversation-forget name`

Beispiel:

```text
Ich heiße Thomas.
Wie heiße ich?
```

Pandora antwortet dann aus dem Conversation Memory.


## MVP 19.0 – Coordinator Agent

Neu:

- CoordinatorAgent
- CoordinatorDecision / CoordinatorResult
- CoordinatorLog
- zentrale Routing-Entscheidung:
  - `memory`
  - `chat`
  - `planner_worker`
- `/user/run` nutzt jetzt den Coordinator
- API:
  - `POST /coordinator/run`
  - `POST /coordinator/decide`
  - `GET /coordinator/logs`
- User-GUI zeigt Coordinator-Entscheidung in den Details

Beispiele:

```text
Hallo
→ route: chat

Ich heiße Thomas.
Wie heiße ich?
→ route: memory

Bitte rechne 2+3*4
→ route: planner_worker
```


## MVP 19.0.1 – Coordinator Details GUI Fix

Fix:

- User-GUI zeigt Coordinator-Details jetzt zuverlässig an.
- `showDetails()` schreibt Route und Decision in `decisionBox`.
- User-GUI nutzt `/coordinator/run`.


## MVP 19.0.2 – Coordinator Details Display Fix

Fix:

- Details-Bereich der User-GUI wird deterministisch aufgebaut.
- `showDetails()` schreibt Route, Reason, Confidence, Provider, Model und Session-ID in `decisionBox`.
- `user.js` wurde sauber neu geschrieben, damit keine alten String-Patches übrig bleiben.


## MVP 19.3 – LLM Reliability Layer

Ziel:

- Lokale LLMs dürfen Pandora nicht mehr durch unsauberes JSON, Markdown-Fences, `<think>`-Blöcke, Reasoning-only-Antworten oder falsche Schemas aus dem Tritt bringen.
- Planner-Fehler durch falsches Modell-JSON werden abgefangen und nachvollziehbar gespeichert.
- Reasoning-Inhalte von LM Studio/Qwen werden für Debugging und spätere Lernschritte abgelegt.

Neu:

- `core/llm_reliability.py`
- `LLMReliabilityLayer`
- `LLMReliabilityReport`
- robuste JSON-Recovery aus:
  - Markdown-Codeblöcken
  - eingebettetem JSON
  - `<think>...</think>` + JSON
- Schema-Recovery für Planner-Antworten wie `{ "result": "14" }`
- `LLMResponse` enthält jetzt:
  - `reasoning`
  - `recovered`
  - `confidence`
  - `reliability`
- Reasoning-Speicher unter `memory/reasoning/<task_type>/`
- Planner-Fallback bei LLM-Schemafehlern
- LM-Studio-Kompatibilität: `response_format` wird nicht mehr standardmäßig gesendet
- Provider-Aliase: `lmstudio`, `lm-studio`, `lm_studio`, `local` → `local_fast`

Beispiel:

```text
Bitte rechne 2+3*4
```

Wenn Qwen fälschlich antwortet:

```json
{"result":"14"}
```

wird daraus eine rekonstruierte Planner-Analyse mit `calculator` als Tool. Falls Recovery nicht möglich ist, nutzt der Planner deterministische Fallback-Regeln und speichert den Fehler in `raw_analysis.llm_analysis_error`.

Tests:

```powershell
python -m pytest
python -m compileall .
```





## MVP 19.3.4 – Single-Pass Capability Gate

Bugfix:

- Coordinator reuses the Tool Development capability decision from `decide()` during `run()`.
- One user request no longer asks the capability-gate LLM twice.
- This avoids duplicate LM Studio calls and prevents a second truncated JSON response from affecting a decision that was already made.
- Tool proposal creation receives the precomputed gap via `precomputed_gap`.

Validation:

- `pytest`: 21 passed
- `compileall`: successful

## MVP 19.3.3 – Deterministic Existing Tool Fast Path

MVP 19.3.3 reduces unnecessary LLM calls for obvious local tool tasks.

Problem:

- `Bitte rechne 2+3*4` triggered the LLM capability gate and then the Planner LLM, although the existing `calculator` tool can handle it deterministically.

Change:

- `ChatResponseRouter.deterministic_existing_tool()` detects conservative known-tool cases such as arithmetic.
- `CoordinatorAgent` routes those tasks directly to `planner_worker` before asking the capability gate.
- `PlannerAgent` skips LLM analysis when a known registered deterministic tool can be selected safely.
- Capability discovery remains LLM-first for ambiguous/missing capabilities such as stock lookup or weather lookup.

Expected behavior:

```text
Bitte rechne 2+3*4
→ no capability-gate LLM call
→ no planner LLM call
→ calculator
→ 14
```

Validation:

- `pytest`: 20 passed
- `compileall`: successful

## MVP 19.3.2 – LLM Capability Gate

MVP 19.3.2 ersetzt das fest verdrahtete Capability-Routing durch eine generische LLM-Entscheidung.

Neue Entscheidungskette:

```text
User request
↓
LLM Capability Gate
↓
Kann direkt antworten?
↓ nein
Ist vorhandenes Tool ausreichend?
↓ nein
tool_needed / Tool Development
```

Wichtig:

- Börsenkurse, Wetter, Dateien, Live-Daten, Gerätezugriff usw. müssen nicht mehr einzeln fest verdrahtet werden.
- Das LLM liefert eine strukturierte `CapabilityDecision`.
- Vorhandene Tools werden berücksichtigt, z.B. `calculator` für Rechenaufgaben.
- Keyword-Erkennung bleibt nur noch transparenter Fallback bei LLM-Ausfall.
- `/tool-development/analyze` akzeptiert jetzt `provider_name`, `model` und `timeout`.

Beispiele:

```text
Bitte rechne 2+3*4
→ existing_tool_sufficient: calculator
→ route: planner_worker

Ich möchte den aktuellen Börsenkurs von BASF abrufen
→ tool_needed: stock_price_lookup
→ route: tool_development
```

Prüfung:

- `pytest`: 18 passed
- `compileall`: erfolgreich

## MVP 19.3.1 – Capability Gap Routing Fix

Fix:

- Requests such as `Ich möchte gerne Wörter zählen.` no longer fall through to normal chat when the `word_count` tool is missing.
- Requests such as `Ich möchte das aktuelle Wetter abrufen.` are recognized as missing capability `weather_lookup`.
- Coordinator route `tool_development` creates a controlled tool proposal instead of producing a friendly chat answer that hides the missing capability.
- Added API endpoints:
  - `POST /tool-development/analyze`
  - `POST /tool-development/propose`

Examples:

```text
Ich möchte gerne Wörter zählen.
→ route: tool_development
→ capability: word_count

Ich möchte das aktuelle Wetter abrufen.
→ route: tool_development
→ capability: weather_lookup
```

Validation:

```powershell
python -m pytest
python -m compileall .
```

Result: 14 tests passed, compileall successful.

## MVP 19.4 – Model Router

Ziel:

- Pandora trennt Alltag/erste Anfrage von aufwendiger Code- und Tool-Erzeugung.
- Normale Aufgaben laufen weiter lokal und schnell.
- Tool-/Code-Erzeugung und Core-Reviews werden zentral auf ein Cloud-Expert-Modell geroutet.
- Agenten sollen nicht selbst hart verdrahten, ob lokal oder Cloud verwendet wird.

Neu:

- `core/model_router.py`
- `ModelRouter`
- zentrale `model_routes` in `config/llm/llm_config.json`
- Provider-Aliase:
  - `lmstudio` → `local_fast`
  - `local` → `local_fast`
  - `cloud`, `cloud_expert`, `chatgpt` → `openai`
- `LLMRouter` delegiert an `ModelRouter`
- API:
  - `GET /model-router/routes`
  - `GET /model-router/route/{purpose}`
- CLI:
  - `python main.py model-routes`
  - `python main.py model-route tool_generation`

Standard-Routing:

```text
chat             -> local_fast / LM Studio
tool_selection   -> local_fast / LM Studio
planning         -> local_fast / LM Studio
reflection       -> local_fast / LM Studio
tool_generation  -> cloud_expert -> openai
core_review      -> cloud_expert -> openai
code_review      -> cloud_expert -> openai
```

Wichtig:

- Explizite Provider-Overrides funktionieren weiterhin, z.B. `--provider mock` oder `provider_name=lmstudio`.
- Ohne `OPENAI_API_KEY` fällt Tool-Generation weiterhin kontrolliert auf den vorhandenen Fallback zurück.
- Cloud darf weiterhin nur Proposals erzeugen. Aktivierung bleibt manuell.

Prüfung:

```powershell
python -m pytest
python -m compileall .
```

Ergebnis: 25 Tests erfolgreich, compileall erfolgreich.


## MVP 19.5.1 – Secure LLM Profiles

Ziel:

- Pandora kann zwischen privatem Cloud-Zugang und Company-LLM wechseln.
- Zugangsdaten und interne Company-Netzadressen landen nicht im Repository.
- Cloud-Routing bleibt zentral über den Model Router steuerbar.

Neu:

- `config/llm/llm_config.template.json` als sichere GitHub-taugliche Vorlage
- `config/llm/llm_config.local.json` als private lokale Override-Datei, nicht für GitHub
- `.env` Unterstützung ohne zusätzliche Abhängigkeit
- `.env.example` mit Platzhaltern
- `.gitignore` schützt `.env` und `*.local.json`
- `LLMConfig` lädt jetzt Template + Legacy Config + Local Override + ENV
- Profile:
  - `private` → `openai` / ChatGPT über `OPENAI_API_KEY`
  - `company` → `company_llm` über `COMPANY_LLM_*`
- `GET /llm/config` und `python main.py llm-config` geben nur redaktierte Konfiguration aus
- `GET /llm/config/security` und `python main.py llm-config-security` prüfen auf versehentliche Inline-Secrets

Private Nutzung:

```powershell
copy memory\llm_config.local.example.json memory\llm_config.local.json
notepad memory\llm_config.local.json
notepad .env
```

Beispiel `config/llm/llm_config.local.json`:

```json
{
  "active_profile": "company"
}
```

Beispiel `.env`:

```env
OPENAI_API_KEY=
COMPANY_LLM_BASE_URL=
COMPANY_LLM_API_KEY=
COMPANY_LLM_MODEL=
```

Wichtig:

- Keine echten Keys in `config/llm/llm_config.template.json`.
- Keine Company-URLs in GitHub-Dateien.
- Secrets nur über `.env` oder echte Prozess-Umgebungsvariablen.

Prüfung:

```powershell
python main.py llm-config-security
python main.py model-route tool_generation
python main.py cloud-expert-status
```

Tests:

```powershell
python -m pytest
python -m compileall -q .
```

## MVP 19.5 – Cloud Expert Provider

Ziel:

- Alltagsgeschäft bleibt lokal und schnell.
- Aufwendige Aufgaben wie Tool-Erzeugung, Code Review und Core Review werden zentral auf ein Cloud-Expert-Modell geroutet.
- Cloud-Ausgaben werden weiterhin nur als Proposal verarbeitet und nie automatisch aktiviert.

Neu:

- `CloudExpert`
- Provider-/ENV-Status für OpenAI-kompatible Cloud Expert Nutzung
- API:
  - `GET /cloud-expert/status`
  - `POST /cloud-expert/smoke`
- CLI:
  - `python main.py cloud-expert-status`
  - `python main.py cloud-expert-smoke`
  - `python main.py cloud-expert-smoke --live --prompt "Sag kurz OK"`
- `LLMTaskType.CODE_REVIEW`
- `code_review` Routing
- Tool-Code-Generierung deaktiviert stillen Mock-Fallback. Wenn `OPENAI_API_KEY` fehlt, fällt Pandora kontrolliert auf deterministische Proposal-Gerüste zurück.

Konfiguration:

```json
{
  "providers": {
    "openai": {
      "type": "openai",
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "default_model": "gpt-4.1-mini"
    }
  },
  "model_routes": {
    "tool_generation": {"provider": "cloud_expert"},
    "core_review": {"provider": "cloud_expert"},
    "code_review": {"provider": "cloud_expert"}
  }
}
```

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "<dein_api_key>"
```

Danach ein neues Terminal öffnen.

Prüfung:

```powershell
python main.py cloud-expert-status
python main.py model-route tool_generation
python main.py tool-generate weather_lookup --no-tests
```

Sicherheitsregel:

Cloud Expert darf Code vorschlagen. Pandora validiert lokal. Aktivierung bleibt manuell.




## MVP 19.5.5 – Configuration Refactoring

Ziel:

- Statische Konfiguration liegt nicht mehr in `memory/`, sondern unter `config/`.
- `memory/` bleibt für Laufzeitdaten, Chatverläufe, Logs, Reasoning und gelernte Fakten reserviert.
- Private Overrides bleiben geschützt und werden nicht versioniert.

Neue Struktur:

```text
config/
├─ llm/
│  ├─ llm_config.json
│  ├─ llm_config.template.json
│  └─ llm_config.local.example.json
├─ tools/
│  ├─ tool_registry.json
│  └─ execution_policy.json
├─ skills/
│  └─ skill_registry.json
└─ system/
   └─ pandora.json
```

Neue Komponente:

- `ConfigManager`

Neue CLI/API:

```powershell
python main.py config-paths
```

```text
GET /config/paths
```

Kompatibilität:

- Legacy-Dateien in `memory/` werden beim Laden weiterhin als Fallback akzeptiert.
- Neue Schreibpfade zeigen aber auf `config/`.

Prüfung:

```powershell
python -m pytest
python -m compileall -q .
```

## MVP 19.5.4 – Capability Gate Chat-Veto Hotfix

Fix:

- Kleine lokale Modelle können eine klare Fähigkeitsanfrage fälschlich als normalen Chat klassifizieren.
- Pandora nutzt weiterhin LLM-first Routing, aber ein transparenter Fallback darf eine klare fehlende Capability gegen eine falsche Chat-Entscheidung absichern.
- Beispiel: `Ich möchte Wörter zählen` wird nicht mehr vom Chat-Fallback beantwortet, wenn `word_count` als fehlende Capability erkennbar ist.
- Der ursprüngliche LLM-Entscheid bleibt in `gap.decision` sichtbar.
- Neue Quelle: `fallback_after_llm_direct_answer`.

Prüfung:

- `pytest`: 45 passed
- `compileall`: erfolgreich

## MVP 19.5.3 – Capability Gate Confidence Normalization

Bugfix:

- Kleine lokale Modelle können eine korrekte Capability-Entscheidung liefern, aber `confidence: 0.0` setzen.
- Pandora verwirft solche strukturell eindeutigen Entscheidungen nicht mehr.
- Beispiel: `Ich möchte Wörter zählen` mit `tool_needed=true` und `capability=word_count` wird jetzt trotz Modell-Confidence `0.0` als Tool-Gap erkannt.
- Die originale Modell-Confidence bleibt in `model_confidence` sichtbar.
- Die von Pandora normalisierte Entscheidungs-Confidence steht in `confidence`.

Prüfung:

- `pytest`: 43 passed
- `compileall`: erfolgreich

## MVP 19.5.2 – Profile Manager & Connectivity Tests

Ziel:

- Zwischen privaten und Firmen-LLM-Umgebungen umschalten, ohne Zugangsdaten oder interne URLs ins Repository zu schreiben.
- Provider-Konfigurationen prüfen, ohne Secrets auszugeben.
- Live-Connectivity bewusst erst mit `--live` ausführen.

Neue CLI-Befehle:

```powershell
python main.py llm-profile-status
python main.py llm-profile private
python main.py llm-profile company
python main.py llm-provider-status cloud_expert
python main.py llm-provider-status company_llm
python main.py llm-provider-smoke cloud_expert
python main.py llm-provider-smoke cloud_expert --live
```

Neue API-Endpunkte:

- `GET /llm/profile/status`
- `POST /llm/profile`
- `GET /llm/provider/status/{provider}`
- `POST /llm/provider/smoke`

Umschalten:

```powershell
python main.py llm-profile private
python main.py llm-profile company
```

Das schreibt nur diese lokale, gitignorierte Datei:

```text
config/llm/llm_config.local.json
```

Secrets bleiben in `.env` oder in echten Umgebungsvariablen:

```env
OPENAI_API_KEY=...
COMPANY_LLM_BASE_URL=...
COMPANY_LLM_API_KEY=...
COMPANY_LLM_MODEL=...
```

Sicherheit:

- Statusausgaben zeigen nur `api_key_present: true/false`.
- Company-URLs aus ENV werden als `<from env>` angezeigt.
- Smoke-Tests laufen ohne `--live` nicht gegen das Netzwerk.
- `.env` und `config/llm/llm_config.local.json` bleiben durch `.gitignore` geschützt.

Validierung:

```powershell
python -m pytest
python -m compileall .
python main.py llm-config-security
```
