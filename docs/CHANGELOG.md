# Changelog

## MVP 28.0 – Cognitive Identity & Self Model

- Added `core/cognitive_identity.py` as a read-only identity and self-model layer.
- Added CLI commands `cognitive-identity-status`, `cognitive-identity-card`, `cognitive-boundaries` and `cognitive-self-model`.
- Added API endpoints under `/api/cognitive/identity/*`.
- Added explicit capability boundaries, truthfulness rules and safe operating statement.
- Added regression tests for read-only guarantees and approval boundaries.

## MVP 27.2 – Adaptive Tool Selection

- Added `core/adaptive_tool_selection.py`.
- Added CLI commands `adaptive-tool-selection-status` and `adaptive-tool-select`.
- Added API endpoints `/api/cognitive/adaptive-tool-selection/status` and `/preview`.
- Added tests for calculator selection, stock-history tool-gap detection and cloud SAFE policy.
- Preserves the rule: no tool execution, no code generation and no registry writes during selection.

# MVP 27.1 – Adaptive Source Selection

- Added `core/adaptive_source_selection.py`.
- Added CLI commands `adaptive-source-selection-status` and `adaptive-source-select`.
- Added API endpoints `/api/cognitive/adaptive-source-selection/status` and `/preview`.
- Added adaptive source ranking based on cognitive plan mode, interpreted intent and profile policy.
- Added source alias normalization while keeping Python-side governance validation.
- Added regression tests for Obsidian/knowledge lookup, tool-proposal source selection and cloud source blocking.

# MVP 25.6 – Cognitive Context Pipeline

- Added `core/cognitive_context_pipeline.py`.
- Added CLI commands `cognitive-pipeline-status` and `cognitive-pipeline-preview`.
- Added API endpoints `/api/cognitive/pipeline/status` and `/api/cognitive/pipeline/preview`.
- Added pipeline trace across request interpretation, capability analysis, Python orchestration and context preparation.
- Added regression tests for Vault context preservation and tool-gap safety.

# Changelog

## MVP 24.9 – Review Scheduler & Manual Run Center

- Added Review Scheduler service for controlled Night Review triggering.
- Added CLI commands for status, manual run, due-run and history.
- Added API endpoints under `/api/review-scheduler/*`.
- Added `/review-scheduler` web page with dark-theme GUI.
- Added `.env.example` scheduler settings.
- Scheduler is not a daemon and performs no automatic action execution.
