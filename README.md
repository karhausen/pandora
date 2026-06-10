# Pandora Agent

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
