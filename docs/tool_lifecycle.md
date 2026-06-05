# Tool Lifecycle Manager

MVP 20.1 ergänzt die Controlled Tool Factory um Verwaltung installierter Tools.

## Status

- `ACTIVE`: Tool kann ausgeführt werden.
- `DISABLED`: Tool bleibt installiert, wird aber nicht ausgeführt.
- `DEPRECATED`: Tool ist veraltet und wird ebenfalls nicht ausgeführt.
- `FAILED`: Tool ist als fehlerhaft markiert.

## CLI

```bash
python3 main.py tool-info <tool_id>
python3 main.py tool-disable <tool_id>
python3 main.py tool-enable <tool_id>
python3 main.py tool-deprecate <tool_id>
python3 main.py tool-uninstall <tool_id>
python3 main.py tool-stats <tool_id>
```

## Aliase

Cloud-generierte Tool-IDs können vom Capability-Namen abweichen, z.B. `word_counter` statt `word_count`.
Beim Installieren speichert Pandora Capability-Namen als Alias, damit beide Namen auflösbar sind.

## Nutzungsstatistik

Pandora schreibt Laufzeitstatistiken nach `memory/tool_usage_stats.json`:

```json
{
  "executions": 1,
  "successes": 1,
  "failures": 0,
  "last_used": "..."
}
```

## Sicherheit

Deaktivierte oder veraltete Tools werden vom `ToolExecutor` abgewiesen, auch wenn sie noch in der Registry stehen.
