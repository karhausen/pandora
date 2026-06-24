# Pandora Dokumentation

Dieser Ordner enthält technische Dokumentation zu Pandora.
Leere Kurzdateien wurden entfernt. Die wichtigsten Einstiege sind jetzt konsolidiert.

## Einstieg

- `overview.md` – Was Pandora ist und wie die Hauptbausteine zusammenhängen
- `configuration.md` – Profile, LLM-Routing, lokale Konfiguration und Secrets
- `commands.md` – wichtige CLI-Befehle
- `gui.md` – verfügbare Web-GUI-Seiten
- `knowledge_base.md` – User Knowledge Base, Metadaten und Governance
- `architecture.md` – ausführlichere Architekturdetails
- `roadmap.md` – Entwicklungsrichtung

## Detaildokumente

Viele MVPs haben eigene Detaildokumente, zum Beispiel:

- `control_core.md`
- `maintenance_manager.md`
- `proposal_approval_workflow.md`
- `llm_routing_editor.md`
- `llm_fallback_diagnostics.md`
- `user_knowledge_base.md`
- `knowledge_search_context_injection.md`
- `knowledge_metadata_governance.md`
- `knowledge_governance_hardening.md`

## Spätere Nutzung in Pandora

Ausgewählte, stabile Dokumente können später nach folgendem Pfad kopiert werden:

```text
user_knowledge/public/pandora/
```

Dann kann Pandora diese Inhalte über Knowledge Search und Context Injection selbst nutzen.

- [Knowledge Editor](knowledge_editor.md)


## Learning Insights

```bash
python main.py learning-insights --rebuild
python main.py learning-insight-status
```
