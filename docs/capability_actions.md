# MVP 23.3.1 – Capability Actions Integration

Capability Actions machen den Capability Graph praktisch nutzbar.

Pandora erzeugt aus priorisierten Capability Gaps prüfbare nächste Schritte:

- `knowledge_candidate`: Wissen fehlt zuerst.
- `tool_candidate`: Wissen ist vorhanden, aber ein Tool fehlt.
- `skill_candidate`: Tool/Wissen existiert, aber kein wiederverwendbarer Skill.
- `knowledge_improvement`: vorhandenes Wissen/Capability-Beziehungen sollten verbessert werden.

## Sicherheitsregel

Capability Actions sind nur Vorschläge. Sie führen keinen Code aus, installieren keine Tools, aktivieren keine Skills und ändern keine Knowledge-Dateien automatisch.

Jede Action wird als JSON unter `proposals/capability_actions/<id>/proposal.json` gespeichert und erscheint dadurch in der Review Inbox.

## CLI

```bash
python main.py capability-actions-status
python main.py capability-actions
python main.py capability-actions-rebuild --limit 50
python main.py capability-action-show <action_id>
```

## API

```text
GET  /api/capabilities/actions
GET  /api/capabilities/actions/status
GET  /api/capabilities/actions/{action_id}
POST /api/capabilities/actions/rebuild
```

## GUI

Im Capability Explorer gibt es den Bereich **Capability Actions**. Dort können Actions erzeugt, angezeigt und anschließend über die Review Inbox geprüft werden.
