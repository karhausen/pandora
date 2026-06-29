# MVP 25.3 – Request Interpreter

Der Request Interpreter ist die erste Schicht des neuen Cognitive Core.

Er beantwortet keine Benutzerfrage direkt. Er analysiert nur semantisch, welche Informationsräume, Tools, Skills oder Capability-Gaps für die Anfrage wahrscheinlich relevant sind.

## Grundregel

```text
LLM empfiehlt.
Python validiert.
Python handelt.
```

Der Interpreter darf niemals:

- Dateien lesen,
- Tools ausführen,
- Policies umgehen,
- finale Entscheidungen treffen,
- Code aktivieren.

## Pipeline

```text
User Request
   ↓
Request Interpreter
   ↓
Python Orchestrator / Governance
   ↓
Context Builder
   ↓
Ranking / Duplicate Removal / Budget
   ↓
Prompt Builder
   ↓
LLM-Antwort
```

## Ergebnisstruktur

Der Interpreter liefert eine strukturierte Empfehlung:

```json
{
  "intent": "knowledge_lookup",
  "source_spaces": ["obsidian_vault", "conversation_memory"],
  "tools": [],
  "skills": [],
  "capability_gaps": [],
  "confidence": 0.65,
  "recommended_next_step": "context_lookup"
}
```

## Bekannte Quellenräume

- `conversation_memory`
- `long_term_memory`
- `user_knowledge`
- `obsidian_vault`
- `capability_graph`
- `learning_engine`
- `tool_registry`
- `skill_registry`

## CLI

```bash
python main.py request-interpreter-status
python main.py request-interpret "Was war meine letzte Notiz?" --provider-name mock
```

## Einordnung

MVP 25.3 ist bewusst noch keine vollständige autonome Steuerung. Es legt die Grundlage dafür, dass spätere Komponenten wie Capability Analyzer, Tool Gap Workflow, Knowledge Gap Workflow und Python Orchestrator auf einer strukturierten semantischen Voranalyse aufbauen können.
