# MVP 21.8 – GUI Approval API

This MVP exposes the Proposal Review Inbox and Proposal Approval Workflow through a GUI-ready FastAPI interface.

## Purpose

The API is the backend contract for Pandora's future user interface:

- show reviewable proposals in one inbox
- show detail information and risk level
- record user decisions
- show the approval audit log

The API does **not** activate tools, install skills, run generated code, modify core files or change credentials.

## Endpoints

```text
GET  /api/gui/approval/dashboard
GET  /api/gui/approval/inbox
GET  /api/gui/approval/inbox/{item_id}
POST /api/gui/approval/inbox/{item_id}/decision
GET  /api/gui/approval/audit
GET  /api/gui/approval/status
```

## Decision payload

```json
{
  "decision": "approve_next_step",
  "note": "Looks safe for the next controlled design step",
  "decided_by": "user"
}
```

Allowed decisions:

```text
approve_next_step
reject
needs_work
defer
reviewed
```

## Safety rules

- GUI approval only records a decision.
- `execution_allowed` is always `false`.
- `activation_performed` is always `false`.
- High/critical risk approval requires a note.
- Real activation must happen in a later, separate controlled workflow.

## Example

```bash
uvicorn core.api:app --reload
curl http://127.0.0.1:8000/api/gui/approval/dashboard
```
