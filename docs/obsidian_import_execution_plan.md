# MVP 23.5.6 – Obsidian Import Execution Plan

Pandora kann Obsidian-Import-Kandidaten jetzt kontrolliert in die interne `user_knowledge/` Knowledge Base übernehmen.

## Sicherheitsregeln

- Obsidian bleibt read-only.
- Import ist nur aus einem vorhandenen Import-Kandidaten möglich.
- Ausführung erfordert vorherigen Status `accepted_for_next_step`.
- Ausführung erfordert explizit `confirm=true` bzw. `--confirm`.
- Ziel ist ausschließlich `user_knowledge/<area>/...`.
- Bestehende Dateien werden standardmäßig nicht überschrieben.
- Nach dem Import wird ein Audit-Eintrag unter `proposals/obsidian_import_executions/` geschrieben.

## CLI

```bash
python main.py obsidian-import-execution-status
python main.py obsidian-import-execution-list
python main.py obsidian-import-plan <candidate_id>
python main.py obsidian-import-execute <candidate_id> --confirm
```

## API

```text
GET  /api/obsidian/import-executions/status
GET  /api/obsidian/import-executions
GET  /api/obsidian/import-candidates/{candidate_id}/execution-plan
POST /api/obsidian/import-candidates/{candidate_id}/execute
```

## GUI

Im Obsidian Vault Bereich kann ein Import-Kandidat geöffnet werden. Dort können Nutzer den Plan prüfen und den Import gezielt ausführen.
