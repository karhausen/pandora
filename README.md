# Pandora Agent MVP 12.0

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


# MVP 12.0 – Controlled Tool Proposal System

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


# MVP 12.0 – Controlled Tool Activation

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


# MVP 12.0 – Agent Capability Expansion

Neu:

- CapabilityExpansionManager
- CapabilityEventLog
- AgentLoop erkennt fehlende Tool-Fähigkeiten
- AgentLoop erzeugt Tool-Proposals automatisch
- keine automatische Aktivierung
- nach Aktivierung kann der Agent das neue Tool nutzen

## Beispiele

```powershell
python main.py agent-run "word count --text eins zwei drei" --provider mock
python main.py capability-events
python main.py tool-proposal-list
python main.py tool-proposal-activate <ID>
python main.py agent-run "word count --text eins zwei drei" --provider mock
```

Erwartung:

1. erster Lauf erzeugt ein Tool-Proposal
2. User aktiviert kontrolliert
3. zweiter Lauf nutzt das neue Generated-Tool


# MVP 12.0 – Capability Workflow

Neu:

- CapabilityWorkflow
- CapabilityWorkflowLog
- Workflow: Propose only
- Workflow: Propose → Activate
- Workflow: Propose → Activate → Retry
- CLI/API für Capability Workflows

## Beispiele

Nur Proposal:

```powershell
python main.py capability-workflow "reverse text --text abc"
```

Proposal + Aktivierung + erneuter Agent-Lauf:

```powershell
python main.py capability-workflow "word count --text eins zwei drei" --activate --retry
```

Logs:

```powershell
python main.py capability-workflows
python main.py capability-workflow-last
```

Wichtig: Auch hier bleibt Aktivierung explizit. Der Agent aktiviert nicht still im normalen `agent-run`.


# MVP 12.0 – Skill Evolution

Neu:

- SkillPatternDetector
- SkillGenerator
- SkillValidator
- SkillProposalManager
- SkillActivationManager
- generische Skill-Step-Ausführung
- CLI/API für Skill-Proposals und Aktivierung

## Beispiele

```powershell
python main.py skill-propose-from-journal
python main.py skill-proposal-list
python main.py skill-proposal-show <ID>
python main.py skill-proposal-activate <ID>
python main.py agent-run "workflow --text hallo agent" --provider mock
```

MVP 12 aktiviert Skills nur explizit. Der Agent erzeugt und nutzt Routinen erst nach kontrollierter Freigabe.


# MVP 13.0 – Learning & Adaptive Strategy

Neu:

- LearningEngine
- StrategyMemory
- ToolSkillRanker
- FailureAnalyzer
- RecommendationEngine
- AdaptivePlanner
- CLI/API für Rankings, Empfehlungen, Fehler und Strategien

## Beispiele

```powershell
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
python main.py agent-run "Bitte rechne 2+3*4" --provider mock
python main.py learn-from-journal
python main.py rankings
python main.py recommendations
python main.py strategies
python main.py failures
```

MVP 13 lernt konservativ: Strategien werden gespeichert, aber überschreiben noch keine Sicherheitsregeln.
