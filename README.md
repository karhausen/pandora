# Pandora Agent MVP 11.2

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


# MVP 11.2 – Controlled Tool Proposal System

Neu:

- CapabilityDetector
- ToolGenerator
- ToolTestGenerator
- ToolValidator
- ToolProposalManager
- Tool-Proposals unter `tool_proposals/`
- generierte Tool-Kandidaten unter `generated_tools/`
- keine automatische Aktivierung

## Beispiele

```powershell
python main.py tool-propose-task "Bitte JSON formatieren"
python main.py tool-propose-capability word_count
python main.py tool-proposal-list
python main.py tool-proposal-show <ID>
python main.py tool-proposal-prepare <ID>
```

`tool-proposal-prepare` kopiert den validierten Kandidaten nach `generated_tools/`, registriert ihn aber bewusst nicht automatisch.

## API

```text
POST /tool-proposals/from-task
POST /tool-proposals/for-capability
GET  /tool-proposals
GET  /tool-proposals/{proposal_id}
POST /tool-proposals/{proposal_id}/prepare-activation
```

## Sicherheitsregel

Neue Tools entstehen zuerst nur als Proposal mit Code, Test und Validierung. Keine automatische Übernahme in die aktive Tool Registry.


# MVP 11.2 – Controlled Tool Activation

Neu:

- ToolActivationManager
- Aktivierung nur für VALIDATED Tool-Proposals
- Kopieren nach `generated_tools/`
- Registrierung in Tool Registry
- Aktivierungs-Testlauf
- Aktivierungs-Log

## Beispiel

```powershell
python main.py tool-propose-capability word_count
python main.py tool-proposal-list
python main.py tool-proposal-activate <ID>
python main.py agent-run "word count --text eins zwei drei" --provider mock
python main.py tool-activation-log
```

Wichtig: Aktivierung ist jetzt kontrolliert möglich, aber nur nach erfolgreicher Proposal-Validierung und Testausführung.
