# Changelog

## MVP 24.9 – Review Scheduler & Manual Run Center

- Added Review Scheduler service for controlled Night Review triggering.
- Added CLI commands for status, manual run, due-run and history.
- Added API endpoints under `/api/review-scheduler/*`.
- Added `/review-scheduler` web page with dark-theme GUI.
- Added `.env.example` scheduler settings.
- Scheduler is not a daemon and performs no automatic action execution.
