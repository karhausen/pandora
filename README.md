# Pandora Agent MVP 10.0

MVP 10 ist der erste echte Agent-Loop.

Neu:
- AgentLoop
- ActionPlanner
- ExecutionContext
- ResultEvaluator
- TaskJournal
- CLI: agent-run, agent-journal, agent-last
- API: /agent/run, /agent/journal, /agent/last

## Beispiele

```powershell
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
python main.py agent-run "uppercase --text hallo agent" --provider mock
python main.py agent-journal
python main.py agent-last
```

Ablauf:

```text
Task → LLM Analyse → ActionPlanner → Tool/Skill → Evaluation → Journal
```

Das LLM führt weiterhin nichts direkt aus. Es schlägt vor, der Core entscheidet.

## LM Studio

Standardprovider ist `local_fast`:

```text
http://localhost:1234/v1
qwen/qwen3-1.7b
```

Für stabile schnelle Tests nutze:

```powershell
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
```
