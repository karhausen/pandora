# MVP 27.4 – Priority Engine

The Priority Engine ranks reviewable cognitive actions across goals and capability gaps.

It evaluates:

- value
- urgency
- effort
- risk
- confidence
- approval need

## Safety

The Priority Engine never executes tools, activates tools, edits knowledge, writes to Obsidian, persists goals or changes the core. It only creates a reviewable priority list.

## CLI

```bash
python main.py priority-engine-status
python main.py priority-prioritize "Pandora sollte Tool- und Core-Verbesserungen priorisieren"
```

## API

```text
GET /api/cognitive/priority-engine/status
GET /api/cognitive/priority-engine/preview?query=...
```

## Role in the Cognitive Core

The component sits after Goal Manager and Central Decision Engine. It helps Pandora decide what should be reviewed first without bypassing approval workflows.
