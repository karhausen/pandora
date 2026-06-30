# Pandora Agent

## MVP 25.7 – Tool Recommendation Workflow

Der Cognitive Core kann erkannte Tool-Gaps jetzt in sichere Tool-Factory-Briefs überführen. Der Workflow erzeugt Schnittstelle, Testanforderungen, Sicherheitsregeln und Review-Schritte, generiert aber keinen Code und aktiviert nichts automatisch.

CLI:

```bash
python main.py tool-recommendation-status
python main.py tool-recommendation-preview "Baue ein Tool für historische Aktienkurse"
```


## MVP 25.6 – Cognitive Context Pipeline

MVP 25.6 verbindet Request Interpreter, Capability Analyzer, Python Orchestrator und Cognitive Context Builder zu einer prüfbaren End-to-End-Pipeline.

Die Pipeline ist ausdrücklich eine Preview-/Trace-Schicht:

- keine Tool-Ausführung
- keine Code-Generierung
- keine Registry-Aktivierung
- keine Core-Änderung
- kein direkter Dateizugriff durch das LLM

Neue Befehle:

```bash
python main.py cognitive-pipeline-status
python main.py cognitive-pipeline-preview "Was war meine letzte Notiz?" --provider-name mock --limit 5
```

Neue API-Endpunkte:

```text
GET /api/cognitive/pipeline/status
GET /api/cognitive/pipeline/preview
```

# Pandora MVP 25.1.3

## MVP 25.5 – Python Orchestrator

MVP 25.5 ergänzt die Cognitive Pipeline um eine Python-seitige Kontrollschicht.
Der Orchestrator nimmt Empfehlungen aus Request Interpreter und Capability Analyzer entgegen, validiert Quellenräume, Tools, Skills, Capability-Gaps und Freigabepflichten und erzeugt daraus einen prüfbaren Plan.

Wichtig: Der Orchestrator führt nichts aus, liest keine Dateien, erzeugt keinen Code und aktiviert keine Tools. Er bereitet nur kontrollierte nächste Schritte vor.

CLI:

```bash
python main.py python-orchestrator-status
python main.py python-orchestrate "Was war meine letzte Notiz?"
```

API:

```text
/api/cognitive/python-orchestrator/status
/api/cognitive/python-orchestrator/preview
```


# Pandora MVP 24.8.1 - Night Review Web Route Fix

Dieses Release basiert auf MVP 24.8 und korrigiert/prüft die Webroute `/night-review`.

Start:

```bash
python main.py api --host 127.0.0.1 --port 8000
```

Dann öffnen:

```text
http://127.0.0.1:8000/night-review
```

# Pandora


## MVP 25.3 – Request Interpreter

Pandora besitzt nun eine erste semantische Cognitive-Core-Schicht. Der `RequestInterpreter` fragt das LLM nicht nach einer Antwort, sondern nach einer strukturierten Empfehlung: Intent, relevante Quellenräume, mögliche Tools/Skills, Capability-Gaps, Confidence und nächster Schritt.

Wichtig: Der Interpreter liest keine Dateien, führt keine Tools aus und trifft keine finale Entscheidung. Python validiert alle Empfehlungen über Governance, Policies und spätere Orchestrierung.

Wichtige Befehle:

```bash
python main.py request-interpreter-status
python main.py request-interpret "Was war meine letzte Notiz?" --provider-name mock
```

Details: `docs/request_interpreter.md`.

## MVP 25.2 – Context Builder Completion

Pandora erweitert den Cognitive Context Builder um Ranking, Duplicate Removal, Budget-/Context-Packing und Diagnosedaten. Der bestehende GUI-Chat-/Obsidian-Pfad aus MVP 25.1.3 bleibt erhalten. Details: `docs/context_builder_completion.md`.

## MVP 24.2 – Learning Feedback Loop

Pandora besitzt jetzt eine observe-only Learning Engine. Sie sammelt Ereignisse aus Action Inbox und Review-Workflows, berechnet Metriken und Patterns, führt aber keine automatischen Änderungen aus.

Wichtige Befehle:

```bash
python main.py learning-status
python main.py learning-collect
python main.py learning-rebuild
python main.py learning-metrics
python main.py learning-patterns
python main.py learning-events-v24
```

GUI: `http://127.0.0.1:8000/learning`
 Agent

Pandora ist ein lokaler, modularer KI-Assistent mit kontrollierter Agentenarchitektur.
Der Core ist die Schaltzentrale: Er nimmt Aufgaben an, entscheidet über Routing, nutzt LLMs, Tools, Skills und Memory, schützt sensible Daten und erzeugt nachvollziehbare Vorschläge für Wachstum.

Aktueller Stand: **MVP 23.0 – Knowledge Editor GUI**

## Was Pandora aktuell kann

- Chat über lokale, private Cloud- oder Company-LLM-Routen
- Profile für `private` und `company`
- LLM Routing Editor in der GUI
- Tool Center GUI
- Skill Center GUI
- Memory Explorer
- User Knowledge Base mit Policy-Regeln
- Knowledge Search & Context Injection
- Knowledge Governance mit Health Score
- Night Mode / Maintenance Manager
- Proposal Review Inbox und Approval Workflow
- Operations Dashboard
- Release Packaging mit Audit gegen Runtime-Artefakte und Secrets

## Grundprinzip

Pandora darf wachsen, aber nicht unkontrolliert.

```text
Core = stabil, kontrollierend, geschützt
Tools = erweiterbar
Skills = erweiterbar
Memory = wachsend
Knowledge Base = vom User gepflegt
GUI = Steuerung und Transparenz
```

Der aktive Core darf nicht automatisch überschrieben werden. Vorschläge werden geprüft, getestet und vom User genehmigt.

## Schnellstart lokal

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
python main.py status
python main.py api --host 127.0.0.1 --port 8000
```

Dann im Browser öffnen:

```text
http://127.0.0.1:8000/
```

Wichtige GUI-Seiten:

```text
/                         User-GUI
/operations               Operations Dashboard
/approval                 Approval Center
/tools-center             Tool Center
/skills-center            Skill Center
/memory-explorer          Memory Explorer
/knowledge-base           User Knowledge Base
/night-mode               Night Mode Dashboard
/llm-profiles             LLM & Profile Center
```

## Docker

```bash
docker compose build
docker compose up
```

Danach:

```text
http://127.0.0.1:8000/
```

## Konfiguration

Standardkonfiguration:

```text
config/llm/llm_config.json
config/tools/tool_registry.json
config/tools/execution_policy.json
config/skills/skill_registry.json
```

Lokale/private Konfiguration:

```text
config/llm/llm_config.local.json
```

Diese Datei wird nicht ins Release-ZIP gepackt.

Beispiel:

```json
{
  "active_profile": "company",
  "model_routes": {
    "chat": {
      "provider": "cloud_expert",
      "model": "company-default-model",
      "reason": "Company model for normal chat."
    }
  }
}
```

Secrets gehören in `.env` oder echte Umgebungsvariablen, niemals ins Repository:

```env
OPENAI_API_KEY=...
COMPANY_LLM_BASE_URL=...
COMPANY_LLM_API_KEY=...
COMPANY_LLM_MODEL=...
```

## LLM Routing

Pandora unterscheidet Aufgabenarten wie:

```text
chat
planning
tool_design
tool_code_generation
code_review
core_review
maintenance
night_mode
```

Die Routen werden im LLM & Profile Center gepflegt:

```text
http://127.0.0.1:8000/llm-profiles
```

Wichtig: Die User-GUI nutzt die zentrale Chat-Route. Es gibt keinen separaten Provider-Override mehr im Chat.

## User Knowledge Base

Eigene Wissensdateien liegen unter:

```text
user_knowledge/
├── public/
├── restricted_cloud_allowed/
└── private_local_only/
```

Bedeutung:

```text
public                    lokal + Cloud erlaubt
restricted_cloud_allowed  Cloud möglich, aber nur nach Policy-Prüfung
private_local_only        nur lokales LLM, niemals Cloud
```

Empfohlenes Format ist Markdown mit YAML-Metadaten:

```markdown
---
title: Tool Factory
tags:
  - pandora
  - tools
visibility: public
cloud_allowed: true
priority: high
last_reviewed: 2026-06-10
---

# Tool Factory

Notizen und Wissen zur Pandora Tool Factory.
```

Governance prüfen:

```bash
python main.py knowledge-governance-run
python main.py knowledge-metadata-audit
```

## Wichtige CLI-Befehle

Status und API:

```bash
python main.py status
python main.py control-status
python main.py heartbeat
python main.py api --host 127.0.0.1 --port 8000
```

Operations und Maintenance:

```bash
python main.py maintenance-status
python main.py maintenance-run --dry-run --force
python main.py operations-preview
python main.py operations-run --force
```

Review und Approval:

```bash
python main.py review-inbox-list
python main.py review-inbox-show <item_id>
python main.py approval-pending
python main.py approval-decide <item_id> --decision approve_next_step --note "geprüft"
python main.py approval-audit
```

Tools und Skills:

```bash
python main.py tool-list
python main.py tool-info <tool_id>
python main.py tool-enable <tool_id>
python main.py tool-disable <tool_id>
python main.py skill-candidate-status
python main.py skill-candidate-run --dry-run --force
```

Knowledge:

```bash
python main.py knowledge-governance-status
python main.py knowledge-governance-run
python main.py knowledge-metadata-audit
```

Tests:

```bash
python -m pytest -q
python -m compileall -q .
```

Release bauen:

```bash
python scripts/export_release.py --skip-tests
```

## Dokumentation

Der Ordner `docs/` enthält technische Detaildokumente.
Der Einstieg ist:

```text
docs/README.md
docs/overview.md
docs/configuration.md
docs/commands.md
docs/gui.md
docs/knowledge_base.md
docs/roadmap.md
```

Kurze Platzhalter-Dokumente wurden entfernt. Später können ausgewählte Inhalte nach `user_knowledge/public/pandora/` übernommen werden, damit Pandora sie aktiv als Kontext nutzen kann.

## Sicherheit

- Keine Secrets im Repository
- Keine lokalen `.local.json` Dateien im Release
- Keine echten User-Knowledge-Inhalte im Release
- Keine Runtime-Artefakte im ZIP
- Private Knowledge bleibt lokal
- Core-Änderungen nur über Review/Approval
- Mock/Fallback-LLM wird in der Ausführung diagnostisch sichtbar gemacht

## Aktueller nächster Architekturpfad

Nach der Dokumentationsbereinigung ist der nächste große fachliche Schritt:

```text
MVP 23.0 – Capability Graph
```

Vorher sinnvoll: echte Knowledge-Dateien pflegen, Governance prüfen und LLM-Routing stabil testen.


## Capability Explorer

Der Capability Explorer ist unter `/capability-explorer` erreichbar und zeigt Beziehungen zwischen Capabilities, Tools, Skills, Knowledge-Dokumenten und Capability Gaps.


## Aktueller UI-Stand

MVP 23.2.1 führt eine klare GUI-Architektur ein: Chat, Knowledge, Capabilities, Operations und Profiles. Neue Seiten sollen künftig einem dieser Bereiche zugeordnet werden.

## Obsidian Vault

Pandora can read an Obsidian vault and export new notes only into `Pandora_Inbox`. Configure it through `.env`; see `docs/obsidian_vault_integration.md`.


## MVP 23.5.6 – Obsidian Import Execution Plan

Obsidian-Import-Kandidaten können jetzt kontrolliert in `user_knowledge/` übernommen werden. Die Ausführung benötigt einen akzeptierten Kandidaten und `--confirm`. Obsidian bleibt read-only.

```bash
python main.py obsidian-import-plan <candidate_id>
python main.py obsidian-import-execute <candidate_id> --confirm
python main.py obsidian-import-execution-list
```


## Learning Insights

```bash
python main.py learning-insights --rebuild
python main.py learning-insight-status
```


## MVP 24.6 – Action Workflow Chains

Die Unified Action Inbox erzeugt bei `accepted_for_next_step` kontrollierte Folge-Actions. Dadurch wandert ein abgeschlossener Schritt in Done, während der nächste sichere Prüfschritt automatisch in der Inbox erscheint. Es wird weiterhin nichts ohne explizite Bestätigung ausgeführt.


## Obsidian Frontmatter Validation

```bash
python main.py obsidian-validate
```

Uses PyYAML and reports malformed YAML as governance warnings instead of crashing indexing.

## MVP 25.4 – Capability Analyzer

MVP 25.4 ergänzt die Cognitive Pipeline um eine Diagnose-Schicht nach dem Request Interpreter.

Der Capability Analyzer erkennt strukturiert:

- Tool Gaps
- Skill Gaps
- Knowledge Gaps
- Core Gaps
- empfohlene Aktionen und Prioritäten

Er führt keine Tools aus, erzeugt keinen Code und aktiviert nichts automatisch. Änderungen bleiben review-, test-, governance- und freigabepflichtig.

CLI:

```bash
python main.py capability-analyzer-status
python main.py capability-analyze "Analysiere historische Aktienkurse" --provider-name mock
```

API:

```text
GET /api/cognitive/capability-analyzer/status
GET /api/cognitive/capability-analyzer/preview?query=...
```

Doku: `docs/capability_analyzer.md`
