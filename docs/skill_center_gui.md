# MVP 22.3 – Skill Center GUI

Das Skill Center ist die Benutzeroberfläche für installierte Skills, Skill-Kandidaten und Aktivierungshistorie.

## Seiten

- `/skills-center` – Skill Center im Browser
- `/api/gui/skills/dashboard` – Übersicht
- `/api/gui/skills` – installierte Skills
- `/api/gui/skills/{skill_id}` – Skill-Details
- `/api/gui/skills/{skill_id}/action` – sichere Statusaktion
- `/api/gui/skills/candidates` – Skill-Kandidaten
- `/api/gui/skills/activation-log` – Aktivierungshistorie

## Sicherheitsprinzip

Das Skill Center installiert oder generiert keine Skills automatisch.
Es darf nur bereits installierte Skills aktivieren/deaktivieren und Kandidaten sichtbar machen.
Die Entscheidung über neue Skills bleibt im Approval Workflow.
