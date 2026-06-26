# MVP 25.1.2 – Obsidian Frontmatter Parser Fix

## Problem

Beim Erzeugen von Obsidian Import Candidates konnte der Vault-Indexer mit folgendem Fehler abbrechen:

```text
AttributeError: 'bool' object has no attribute 'append'
```

Ursache war der leichte Frontmatter-Parser in `core/obsidian_vault.py`. Er hat YAML-Listen blind an den zuletzt gelesenen Key angehängt. Wenn dieser Key vorher als Boolean erkannt wurde, führte ein Listen-Eintrag zu einem Crash.

## Lösung

`_extract_frontmatter()` wurde robuster gemacht:

- Boolean-Felder bleiben Booleans.
- Listen-Felder werden nur erweitert, wenn der aktuelle Wert wirklich eine Liste ist.
- Inline-Listen wie `tags: [pandora, obsidian]` werden unterstützt.
- Fehlerhafte/streunende Listenzeilen werden ignoriert, statt den gesamten Indexlauf zu blockieren.

## Test

Neuer Regressionstest:

```text
tests/test_mvp25_1_2_obsidian_frontmatter_parser.py
```

Geprüft wird:

- gemischte Boolean- und Listen-Felder
- tolerantes Verhalten bei fehlerhaften Listenzeilen nach einem Boolean-Feld
