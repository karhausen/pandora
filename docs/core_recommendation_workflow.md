# MVP 25.9 – Core Recommendation Workflow

The Core Recommendation Workflow converts validated core gaps into reviewable architecture proposals.

It does **not** modify source files, change policies, build releases, or activate behavior.

## Position in the Cognitive Architecture

```text
Request Interpreter
  ↓
Capability Analyzer
  ↓
Python Orchestrator
  ↓
Core Recommendation Workflow
  ↓
Architecture Review / User Approval
  ↓
Future MVP Implementation
```

## CLI

```bash
python main.py core-recommendation-status
python main.py core-recommendation-preview "Verbessere Pandoras Cognitive Pipeline"
```

## API

```text
GET /api/cognitive/core-recommendation/status
GET /api/cognitive/core-recommendation/preview?query=Verbessere%20Pandora
```

## Safety Guarantees

- no source edits
- no release builds
- no policy changes
- no automatic activation
- user approval required before implementation
- regression tests required for core-impacting changes

## Output

The workflow emits `core_improvement_briefs` containing:

- proposal type
- affected modules
- architecture principles
- impact analysis
- review workflow
- quality checks
- release requirements
