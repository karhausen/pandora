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
