# Pandora Agent

## MVP 28.1 – Cognitive Identity & Self Model

Pandora besitzt jetzt ein explizites, auslesbares Selbstmodell. Die neue Schicht beschreibt Name, Mission, Fähigkeiten, Grenzen, bekannte Schwächen und sichere Betriebsregeln. Sie ist bewusst read-only: keine Tool-Ausführung, keine Aktivierung, keine Obsidian-Schreiboperationen und keine Core-Änderungen.

Neue Befehle:

```bash
python main.py cognitive-identity-status
python main.py cognitive-identity-card
python main.py cognitive-boundaries
python main.py cognitive-self-model
python main.py cognitive-self-model "Baue ein neues Tool für Wetterdaten"
```

Neue API-Endpunkte:

```text
GET /api/cognitive/identity/status
GET /api/cognitive/identity/card
GET /api/cognitive/identity/boundaries
GET /api/cognitive/identity/self-model?query=...
```

---

## MVP 27.1 – Adaptive Source Selection

MVP 27.1 ergänzt Pandora um eine adaptive Quellenauswahl vor dem eigentlichen Context Builder. Der Cognitive Plan darf passende Informationsräume empfehlen; Python normalisiert, priorisiert und validiert diese Empfehlungen gegen Profil- und Governance-Regeln.

Neue Befehle:

```bash
python main.py adaptive-source-selection-status
python main.py adaptive-source-select "Was war meine letzte Notiz?"
```

Neue API-Endpunkte:

```text
GET /api/cognitive/adaptive-source-selection/status
GET /api/cognitive/adaptive-source-selection/preview?query=...
```

Wichtig: Diese Stufe liest keine Dateien, führt keine Tools aus und erzeugt keine Antwort.

---


## MVP 26.0 – Working Memory Foundation

MVP 26.0 ergänzt Pandora um ein temporäres Working Memory für aktive Aufgaben. Es hält Ziele, Hypothesen, Zwischenergebnisse, offene Fragen, Prioritäten, Entscheidungen und nächste Aktionen fest, ohne automatisch in Long-Term Memory, Obsidian oder die Knowledge Base zu schreiben.

Neue Befehle:

```bash
python main.py working-memory-status
python main.py working-memory-preview "Was war meine letzte Notiz?"
```

Neue API-Endpunkte:

```text
GET /api/cognitive/working-memory/status
GET /api/cognitive/working-memory/preview?query=...
```

---

## MVP 25.9 – Core Recommendation Workflow

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


### MVP 26.0 – Working Memory Foundation

MVP 26.0 ergänzt Pandora um ein temporäres Working Memory für aktive Aufgaben. Es hält Ziele, Hypothesen, Zwischenergebnisse, offene Fragen, Prioritäten, Entscheidungen und nächste Aktionen fest, ohne automatisch in Long-Term Memory, Obsidian oder die Knowledge Base zu schreiben.

Neue Befehle:

```bash
python main.py working-memory-status
python main.py working-memory-preview "Was war meine letzte Notiz?"
```

Neue API-Endpunkte:

```text
GET /api/cognitive/working-memory/status
GET /api/cognitive/working-memory/preview?query=...
```

---

## MVP 25.9 – Core Recommendation Workflow

MVP 25.9 ergänzt die Cognitive Architecture um einen kontrollierten Workflow für Core- und Architekturverbesserungen. Erkannte Core-Gaps werden nicht automatisch umgesetzt, sondern als reviewbare Core-Improvement-Briefs vorbereitet.

Neu:

- `core/core_recommendation_workflow.py`
- CLI: `core-recommendation-status`
- CLI: `core-recommendation-preview`
- API: `/api/cognitive/core-recommendation/status`
- API: `/api/cognitive/core-recommendation/preview`
- Doku: `docs/core_recommendation_workflow.md`

Sicherheitsgarantie:

- keine Source-Edits
- keine Policy-Änderungen
- keine Release-Builds
- keine automatische Aktivierung
- User-Freigabe vor Umsetzung

### MVP 25.8 – Knowledge Recommendation Workflow

Pandora kann erkannte Wissenslücken jetzt in reviewbare Knowledge-Improvement-Briefs überführen. Der Workflow schreibt bewusst nicht in Obsidian, User Knowledge oder Memory, sondern erzeugt nur prüfbare Vorschläge mit Source Requirements, Proposal Contract, Review Workflow und Quality Checks.

Neue Befehle:

```bash
python main.py knowledge-recommendation-status
python main.py knowledge-recommendation-preview "Die Dokumentation fehlt für den Cognitive Layer"
```

Neue API-Endpunkte:

```text
GET /api/cognitive/knowledge-recommendation/status
GET /api/cognitive/knowledge-recommendation/preview?query=...
```


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


## MVP 26.1 – Central Decision Engine

MVP 26.1 ergänzt Pandora um eine zentrale Entscheidungsstelle für den Cognitive Layer.

Die Central Decision Engine sammelt:

- Request Interpreter
- Capability Analyzer
- Python Orchestrator
- Tool Recommendation Workflow
- Knowledge Recommendation Workflow
- Core Recommendation Workflow
- Working Memory

und erzeugt daraus ein einziges `central_decision` Objekt.

Sie führt nichts aus, generiert keinen Code und verändert keinen Core. Sie entscheidet nur den nächsten kontrollierten Schritt und ob Benutzerfreigabe nötig ist.

CLI:

```bash
python main.py central-decision-status
python main.py central-decide "Baue ein Tool für historische Aktienkurse"
```

API:

```text
GET /api/cognitive/central-decision/status
GET /api/cognitive/central-decision/preview?query=...
```

Doku: `docs/central_decision_engine.md`


## MVP 26.2 – Approval Interaction Workflow

Der Approval Interaction Workflow macht aus einer Central Decision eine einfache User-Frage.
Pandora fragt nicht nach internen Details, sondern nur an echten Freigabepunkten:

- Wir brauchen Tool XY. Soll ich den Vorschlag ausarbeiten?
- Ich sehe eine Core-Verbesserung. Soll ich einen prüfbaren Vorschlag ausarbeiten?
- Ich sehe eine Wissenslücke. Soll ich einen Knowledge-Vorschlag ausarbeiten?

Befehle:

```bash
python main.py approval-interaction-status
python main.py approval-interaction-preview "Baue ein Tool für historische Aktienkurse"
python main.py approval-interaction-preview "Baue ein Tool für historische Aktienkurse" --user-decision ja
```

Sicherheitsgrenze: Der Workflow erzeugt keinen Code, schreibt keine Dateien, aktiviert keine Tools und ändert keinen Core.


## MVP 26.3 - Proposal Review Loop

MVP 26.3 ergänzt den kontrollierten Review-Schritt nach der Benutzerfreigabe.

Der Ablauf ist bewusst einfach:

1. Pandora erkennt einen Vorschlagsbedarf.
2. Der Benutzer erlaubt die Ausarbeitung.
3. Ein Vorschlag wird als Review Package vorgelegt.
4. Der Benutzer entscheidet: `passt`, `nachbessern` oder `ablehnen`.
5. Erst nach `passt` darf der Vorschlag in den nächsten kontrollierten Gate-Workflow.

Neue Befehle:

```bash
python main.py proposal-review-loop-status
python main.py proposal-review-loop-preview "Baue ein Tool für historische Aktienkurse"
python main.py proposal-review-loop-preview "Baue ein Tool für historische Aktienkurse" --payload-json '{"purpose":"Demo"}' --review-decision passt
```

Sicherheitsregel: Dieser Workflow erzeugt keinen Code, schreibt keine Dateien, aktiviert keine Tools und verändert nicht den Core.


## MVP 26.4 – Proposal Execution Gate

MVP 26.4 adds the final controlled gate after a user-reviewed proposal.
It checks final execution approval plus required test/audit evidence and then
prepares a handoff to the downstream activation, write or release workflow.
It never activates tools, writes knowledge, changes core code or creates releases.

CLI:

```bash
python main.py proposal-execution-gate-status
python main.py proposal-execution-gate-preview "Baue ein Tool für historische Aktienkurse" --payload-json '{"purpose":"...","python_code":"..."}' --review-decision passt --execution-decision aktivieren --test-ok --audit-ok
```

## MVP 26.5 - Cognitive Integration & Regression Hardening

MVP 26.5 verbindet die Cognitive-Komponenten als nachvollziehbaren Preview- und Regression-Flow.

Neu:

- `core/cognitive_integration_regression.py`
- CLI: `cognitive-integration-status`
- CLI: `cognitive-integration-preview`
- CLI: `cognitive-regression-run`
- API: `/api/cognitive/integration/status`
- API: `/api/cognitive/integration/preview`
- API: `/api/cognitive/regression/run`
- Doku: `docs/cognitive_integration_regression.md`

Der MVP führt nichts aus: keine Tool-Ausführung, keine Codegenerierung, keine Knowledge-Writes, keine Tool-Aktivierung, keine Core-Änderung.



## MVP 26.6 - GUI Decision Inbox

MVP 26.6 macht die Central Decision Engine in der GUI nutzbar.

Neue Oberfläche:

```text
/decision-inbox
```

Neue CLI-Befehle:

```bash
python main.py gui-decision-inbox-status
python main.py gui-decision-inbox-preview "Ich brauche ein Tool fuer Aktienkurse"
python main.py gui-decision-inbox-preview "Ich brauche ein Tool fuer Aktienkurse" --user-action ja
```

Neue API-Endpunkte:

```text
GET /api/cognitive/gui-decision-inbox/status
GET /api/cognitive/gui-decision-inbox/preview?query=...
```

Die Decision Inbox zeigt einfache Benutzerentscheidungen:

- Vorschlag ausarbeiten
- Später prüfen
- Ablehnen
- sicher fortfahren

Sicherheitsregel: Die GUI führt nichts aus, generiert keinen Code, aktiviert keine Tools, schreibt kein Wissen und ändert keinen Core. Sie erzeugt nur den nächsten kontrollierten Handoff.

## MVP 27.0 – Cognitive Planning Engine

MVP 27.0 fuehrt eine Planungsstufe vor Antwort oder Aktion ein.

Neue Befehle:

```bash
python main.py cognitive-planning-status
python main.py cognitive-plan "Was war meine letzte Notiz?"
```

Neue API:

```text
GET /api/cognitive/planning/status
GET /api/cognitive/planning/preview?query=...
```

Die Engine erzeugt einen reviewbaren Plan und fuehrt nichts aus.


## MVP 27.2 – Adaptive Tool Selection

Pandora kann benoetigte Tools adaptiv empfehlen und Tool-Gaps erkennen, ohne Tools auszufuehren oder Code zu generieren.

## MVP 27.4 – Goal Manager

MVP 27.4 ergänzt einen konservativen Goal Manager. Er erzeugt aus Anfrage, Cognitive Plan und Central Decision Engine reviewpflichtige Zielkandidaten für Tool-, Knowledge-, Core-, Governance- und Planning-Themen.

CLI:

```bash
python main.py goal-manager-status
python main.py goal-propose "Ich brauche ein Tool fuer Aktienkurse"
```

API:

- `/api/cognitive/goal-manager/status`
- `/api/cognitive/goal-manager/preview`

Sicherheit: Der Goal Manager speichert nichts dauerhaft, führt nichts aus, aktiviert keine Tools und verändert keinen Core.


## MVP 27.4 – Priority Engine

Prioritizes cognitive goals and capability-gap recommendations by value, urgency, effort, risk and confidence. It creates reviewable priority items only; it does not execute tools, persist goals, change knowledge or modify the core.

CLI:

```bash
python main.py priority-engine-status
python main.py priority-prioritize "Pandora sollte Tool- und Core-Verbesserungen priorisieren"
```

## MVP 27.5 – Weekly/Monthly Review Cycles

Neu ist die Review Cycle Engine für kontrollierte Wochen- und Monatsreviews des Cognitive Core.

```bash
python main.py review-cycle-status
python main.py review-cycle-preview "Pandora soll Tools, Wissen und Core regelmäßig verbessern" --cadence weekly
python main.py review-cycle-preview "Pandora Monatsreview" --cadence monthly
```

Die Engine erzeugt nur Review-Pakete und Freigabepunkte. Sie führt nichts aus, schreibt nicht in den Vault, aktiviert keine Tools und verändert keinen Core-Code.


## MVP 27.6 – Cognitive Dashboard Integration

Neu in 27.6:

- Zentrales Cognitive Dashboard für Entscheidungen, Ziele, Prioritäten, Reviews und Working Memory
- GUI: `/cognitive-dashboard`
- CLI: `cognitive-dashboard-status`, `cognitive-dashboard-preview`
- API: `/api/cognitive/dashboard/status`, `/api/cognitive/dashboard/preview`
- Strikt read-only: keine Tool-Ausführung, keine Aktivierung, keine Vault-/Knowledge-/Core-Änderung

Beispiel:

```bash
python main.py cognitive-dashboard-preview "Prüfe den aktuellen Stand von Pandora"
```


## MVP 27.7 – Review-to-Action Workflow

Neu in 27.7:

- Review-Ergebnisse werden zu einfachen Aktionskarten
- User-Aktionen: `Vorschlag ausarbeiten`, `später`, `ablehnen`
- Integration mit Approval Interaction Workflow und Proposal Review Loop
- CLI: `review-to-action-status`, `review-to-action-preview`
- API: `/api/cognitive/review-to-action/status`, `/api/cognitive/review-to-action/preview`
- Handoff-only: keine Ausführung, keine Aktivierung, keine Core-/Vault-/Knowledge-Änderung

Beispiele:

```bash
python main.py review-to-action-status
python main.py review-to-action-preview "Pandora Weekly Review"
python main.py review-to-action-preview "Pandora Weekly Review" --user-action ja
python main.py review-to-action-preview "Pandora Weekly Review" --user-action später
```

### MVP 27.8 – Action Proposal Handoff

- Neue Komponente `core/action_proposal_handoff.py`
- Übergibt bestätigte Review-to-Action Karten an Tool-/Knowledge-/Core-Proposal-Flows
- CLI: `action-proposal-handoff-status`, `action-proposal-handoff-preview`
- API: `/api/cognitive/action-proposal-handoff/status` und `/preview`
- Doku: `docs/action_proposal_handoff.md`
- Sicherheitsgarantie: kein Code, keine Ausführung, keine Persistenz, keine Core-Änderung
