# MVP 30.1 – Unified Capability Model

Basis: MVP 30.0 – No Keyword Routing

## Ziel

Pandora soll nicht mehr in Sonderfällen wie Tool, Skill, Memory, Vault oder Workflow denken. Diese Quellen werden als neutrale `CapabilityRecord`-Objekte beschrieben und dem LLM als einheitliche Fähigkeitsliste übergeben.

## Änderungen

- Neues Modell: `core/capability_model.py`
  - `CapabilityRecord`
  - neutrale Felder: `id`, `name`, `kind`, `description`, `status`, `security_level`, `permissions`, `provider`, `implementation_ref`, `reliability`
- `CapabilitySnapshot` erweitert:
  - neue Hauptliste `capabilities`
  - bestehende Felder `tools`, `skills`, `knowledge_sources`, `memory_sources`, `workflows` bleiben für Kompatibilität erhalten
- `CapabilitySnapshotBuilder` erzeugt CapabilityRecords für:
  - Tools
  - Skills
  - Knowledge Sources
  - Memory Sources
  - Workflows
- `CapabilityOrchestrator` angepasst:
  - Prompt verweist jetzt explizit auf `snapshot.capabilities`
  - LLM kann `needed_capabilities` zurückgeben
  - Python validiert unbekannte angeforderte Capability-IDs
- Tests ergänzt:
  - Snapshot enthält einheitliche Capabilities
  - CapabilityRecord ist neutral und LLM-lesbar
  - Orchestrator nutzt `snapshot.capabilities`

## Bewusst nicht enthalten

- Keine Sidebar / Live-Console
- Keine neue GUI
- Keine großen Änderungen an Tool-Ausführung oder Knowledge-Index
- Keine Rückkehr zu Keyword-Routing

## Test

```text
6 passed
```
