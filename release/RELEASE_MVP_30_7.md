# Release MVP 30.7 – LLM Conversation Loop

## Ziel

Pandora nutzt eine explizite, begrenzte LLM-Conversation-Loop: Das LLM wählt pro Runde eine Route, Python dispatcht nur, Context wird gesammelt, und die finale Antwort entsteht erst im finalen LLM-Pass.

## Enthalten

- Route-Loop als Conversation Loop dokumentiert und in der Ausführung ausgewiesen
- `PANDORA_LLM_CONVERSATION_MAX_ROUNDS` für maximale Rundenanzahl
- Terminal-Routen: `direct_answer`, `clarify_user`
- Guard gegen wiederholte gleiche Quellenroute
- Release-Dokumente nach `release/` verschoben

## Weiterhin deaktiviert

- Tools
- Skills
- Capability Gap
- Tool Factory
- Evolution
