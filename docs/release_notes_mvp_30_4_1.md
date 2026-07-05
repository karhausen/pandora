# MVP 30.4.1 – Chat Session API Compatibility Fix

Basis: MVP 30.4 – Knowledge Routing Stabilization

## Ziel

Der Knowledge-Routing-Umbau aus 30.4 bleibt unverändert. Dieser Hotfix stellt nur die API-Kompatibilität für die GUI-Chat-Sessions wieder her.

## Fix

`core/chat_service.py` enthält wieder:

- `create_session(title)`
- `list_sessions()`
- `get_session(session_id)`
- `delete_session(session_id)`

## Nicht geändert

- Kein Tool-Routing
- Keine Tool-Factory
- Keine Capability-Gap-Logik
- Kein Planner/Worker im Chat-Hauptpfad
- Kein Architekturumbau

## Tests

```text
25 passed
```
