# MVP 25.6 – Cognitive Context Pipeline

The Cognitive Context Pipeline connects the cognitive components introduced in MVP 25.3–25.5 into one auditable preview flow.

## Purpose

Pandora should not let the LLM directly read files, execute tools or make final decisions. The pipeline separates semantic interpretation from deterministic validation and context preparation.

## Flow

```text
User Request
  ↓
Request Interpreter
  ↓
Capability Analyzer
  ↓
Python Orchestrator
  ↓
Context Builder
  ↓
Ranking
  ↓
Duplicate Removal
  ↓
Budget
  ↓
Prompt Context Ready
```

## Guarantees

- no tool execution
- no code generation
- no registry activation
- no core modification
- no direct LLM file access
- Python validates recommendations before action

## CLI

```bash
python main.py cognitive-pipeline-status
python main.py cognitive-pipeline-preview "Was war meine letzte Notiz?" --provider-name mock --limit 5
```

## API

```text
GET /api/cognitive/pipeline/status
GET /api/cognitive/pipeline/preview?query=Was%20war%20meine%20letzte%20Notiz%3F
```

## Release intent

MVP 25.6 does not replace the GUI chat flow. It provides a traceable preview layer that proves the cognitive chain can be inspected end-to-end before later MVPs use it for controlled action workflows.
