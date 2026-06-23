# MVP 23.5.7 – Obsidian Capability Graph Integration

Pandora kann Obsidian-Notizen jetzt als optionale, read-only Quelle in den Capability Graph aufnehmen.

## Was wird übernommen?

- Markdown-Dateipfad
- Titel
- Tags (`#tag`)
- Wikilinks (`[[Link]]`)
- Wortanzahl
- Änderungszeitpunkt
- Hash

## Was wird nicht gemacht?

- Keine Änderung am Obsidian Vault
- Kein Verschieben
- Kein Löschen
- Kein Überschreiben
- Kein automatischer Import in `user_knowledge/`

## Beziehungen im Graph

```text
capability -> obsidian_note
```

Relation:

```text
has_obsidian_note
```

Tags und Wikilinks werden zu Capability-Kandidaten. Beispiel:

```markdown
# Spektrumanalyse

#funktechnik #messtechnik
Siehe [[Kalibrierung]]
```

führt zu Beziehungen wie:

```text
funktechnik -> Obsidian Note
messtechnik -> Obsidian Note
Kalibrierung -> Obsidian Note
```

## Bedienung

```bash
python main.py obsidian-status
python main.py obsidian-index
python main.py capability-rebuild
python main.py capability-show funktechnik
```

Im Capability Explorer erscheinen Obsidian-Notizen als verbundene Wissensquelle.
