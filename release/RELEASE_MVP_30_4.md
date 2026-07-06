# MVP 30.4 – LLM-Led Route Registry

Ziel: Der Router trifft keine fachliche Entscheidung mehr aus dem User-Text.

## Kernprinzip

```text
User
  -> Prompt Builder
  -> LLM waehlt Route
  -> Router validiert und dispatcht
  -> Route holt Kontext / fuehrt erlaubte Aktion aus
  -> LLM formuliert Antwort
  -> Router liefert Antwort an User
```

## Aktive Routen in diesem MVP

- `direct_answer`
- `vault_search`
- `memory_search`
- `clarify_user`

## Bewusst deaktiviert

- `tool_execute`
- `skill_execute`
- `capability_gap`
- Tool Factory
- Planner/Worker im Chat-Hauptpfad
- Tool Development im Chat-Hauptpfad

## Wichtige Regel

Der Router besitzt nur eine Registry und Handler. Er entscheidet nicht anhand von Keywords, ob Vault, Memory, Tool oder Chat benutzt werden soll.

## Tests

```text
23 passed
```
