# MVP 22.0 – Operations Dashboard

The Operations Dashboard gives Pandora a practical control room for daily use.
It combines core status, maintenance readiness, review inbox metrics and approval
status in one browser page.

## Routes

- `GET /operations` – browser dashboard
- `GET /api/gui/operations/dashboard` – aggregated status
- `POST /api/gui/operations/maintenance/preview` – dry-run maintenance plan
- `POST /api/gui/operations/maintenance/run` – controlled maintenance run

## Safety boundary

The dashboard does **not**:

- modify core source files
- install or activate tools
- install or activate skills
- change credentials or profiles
- bypass the approval workflow

Maintenance runs may write reports and proposal packages only. Follow-up actions
still require explicit approval.
