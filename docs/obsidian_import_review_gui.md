# MVP 23.5.8 – Obsidian Import Review GUI

Die Obsidian Import Review GUI macht die sichere Import-Strecke im Browser bedienbar.

## Seite

```text
/obsidian-import-review
```

## Funktionen

- Import-Kandidaten anzeigen und filtern
- Kandidaten aus dem Vault erzeugen
- Quellvorschau anzeigen
- Zielbereich und Zielpfad prüfen
- vorgeschlagene Metadaten anzeigen
- Konflikte und Warnungen aus dem Import-Plan anzeigen
- Entscheidung speichern: `accepted_for_next_step`, `reviewed`, `needs_work`, `deferred`, `rejected`
- Import-Plan aktualisieren
- Import nach `user_knowledge/` explizit ausführen
- letzte Import-Audits anzeigen

## Sicherheitsregeln

- Obsidian bleibt read-only.
- Import schreibt nur nach `user_knowledge/`.
- Import benötigt einen akzeptierten Kandidaten.
- Import benötigt explizite Bestätigung.
- Kein Überschreiben ohne ausdrückliche Auswahl.
