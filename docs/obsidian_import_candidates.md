# MVP 23.5.5 – Obsidian Knowledge Import Candidates

Pandora kann Obsidian-Notizen als Import-Kandidaten für die interne `user_knowledge/` Knowledge Base vorschlagen.

Wichtig: Dieser Schritt importiert nichts automatisch. Es werden nur prüfbare Vorschläge unter `proposals/obsidian_import_candidates/` erzeugt.

## Zweck

Der Obsidian Vault bleibt die externe Wissensquelle. Pandora kann daraus erkennen:

- Welche Notizen könnten in die Pandora Knowledge Base übernommen werden?
- Welcher Zielbereich ist sinnvoll?
- Welche Metadaten wären passend?
- Welche Tags und Wikilinks sprechen für einen Import?

## Sicherheitsregeln

Pandora darf in diesem MVP:

- Obsidian lesen
- Import-Kandidaten erzeugen
- Review-Status speichern

Pandora darf nicht:

- Obsidian-Notizen verändern
- Obsidian-Notizen löschen
- automatisch in `user_knowledge/` schreiben
- Dateien automatisch verschieben

## CLI

```bash
python main.py obsidian-import-candidates-status
python main.py obsidian-import-candidates-build --query Funktechnik
python main.py obsidian-import-candidates-list
python main.py obsidian-import-candidate-show <candidate_id>
python main.py obsidian-import-candidate-mark <candidate_id> --decision accepted_for_next_step --note "passt"
```

## GUI

Im Bereich `/obsidian-vault` gibt es jetzt den Abschnitt **Import-Kandidaten**.

Dort kannst du:

- Kandidaten erzeugen
- Kandidaten filtern
- Details ansehen
- vorgeschlagene Metadaten prüfen
- Entscheidung speichern

## Review Inbox

Die zentrale Review Inbox scannt jetzt zusätzlich:

```text
proposals/obsidian_import_candidates/
```

Dadurch landen die Vorschläge im bestehenden Review/Approval-Konzept.
