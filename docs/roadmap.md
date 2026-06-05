# Pandora Roadmap

## Erledigt bis MVP 19.7

- lokale GUI und Admin-Dashboard
- Coordinator / Planner / Worker
- Conversation Memory und Memory Recall
- LLM Reliability Layer
- Capability Gate
- Model Router
- Cloud Expert Provider und Profile
- sichere Config-Struktur unter `config/`
- Real Tool Design Agent

## Nächste Schritte

### MVP 19.7 – Cloud Tool Code Generator

Cloud Expert erzeugt Code und Tests auf Basis von `tool_design.json`.

### MVP 19.8 – Tool Review Agent

Zweite Prüfung von Design, Code und Tests. Fokus: Sicherheit, Minimalität, Abhängigkeiten.

### MVP 20.0 – Controlled Tool Factory

Kompletter kontrollierter Ablauf:

```text
Capability Gap → Design → Code → Tests → Review → Sandbox → Proposal → Manual Activation
```

### Danach

- semantisches Memory
- Capability Graph
- Docker-Betrieb
- Geräte-/Messgeräte-Tools


## MVP 20.1 – Tool Lifecycle Manager

Installierte Tools können verwaltet, deaktiviert, reaktiviert, deprecatiert, deinstalliert und über Nutzungsstatistiken bewertet werden.


## MVP 20.2 – Tool Factory GUI Workflow

Tool-Proposals können in der User-GUI geprüft, approved, installiert oder rejected werden.


## MVP 20.3 – Tool Quality Gate & Semantic Validation

Semantic validation closes the gap between generated code, tests, and ToolDesign contracts.


## MVP 20.4 – Generic Capability Gap Detection

Status: umgesetzt.

Pandora erkennt generische Capability-Lücken für Live-Daten-/Abruf-Aufgaben wie Aktienkurse oder Wechselkurse, auch wenn sie nicht explizit in der alten Keyword-Liste standen. Prime-/Rechen-/Wissensfragen bleiben lokale Antworten oder vorhandene Tools.
