# MVP 23.4 – Capability Actions UI & Workflow Polish

Ziel: Capability Actions sind nicht nur JSON-Vorschläge, sondern im Alltag prüfbar und bedienbar.

## Neu

- Capability Explorer filtert Actions nach Typ, Priorität, Status und Suchtext.
- Actions können direkt im Capability Explorer auf `accepted_for_next_step` oder `deferred` gesetzt werden.
- Entscheidungen schreiben ausschließlich `review_state.json` neben den Proposal-Daten.
- Es werden keine Tools installiert, keine Skills aktiviert und kein Knowledge automatisch verändert.
- API und CLI bieten Dashboard, Filter und Decision-Endpunkte.

## CLI

```bash
python main.py capability-actions-dashboard
python main.py capability-actions --priority high --action-type tool_candidate
python main.py capability-action-decide "<action_id>" --decision deferred --note "Später prüfen"
```

## API

```text
GET  /api/capabilities/actions/dashboard
GET  /api/capabilities/actions?query=rf&priority=high
POST /api/capabilities/actions/{action_id}/decision
```

## Sicherheitsregel

Auch eine angenommene Action erlaubt nur den nächsten Review-Schritt. Ausführung, Tool-Installation oder Skill-Aktivierung braucht weiterhin einen separaten Workflow.
