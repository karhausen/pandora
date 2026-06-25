# Web-GUI

Start:

```bash
python main.py api --host 127.0.0.1 --port 8000
```

## Seiten

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
/admin                    Admin Dashboard
```

## User-GUI

Die User-GUI ist der Alltagseinstieg.
Sie nutzt die zentrale Chat-Route aus dem LLM Routing Editor und besitzt keinen eigenen Provider-Override mehr.

## Operations Dashboard

Zeigt Status, Maintenance Preview und Wartungsaktionen.

## Approval Center

Zentrale Prüfung von Vorschlägen. Die GUI schreibt Entscheidungen, führt aber keine kritischen Core-Änderungen direkt aus.

## LLM & Profile Center

Zeigt Profile, Providerstatus und Routing-Regeln. Routing-Änderungen werden validiert, gespeichert und auditiert.

## Knowledge Base

Zeigt User-Wissen, Metadaten, Governance-Status und Context-Injection-Preview.
