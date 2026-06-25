# Nightly Governance Review

Pandora darf nachts analysieren, zusammenfassen und Vorschläge vorbereiten. Pandora darf dabei keine Core-Dateien ändern, keine Tools/Skills aktivieren und keine externen Aktionen starten.

## Ziel

Der Review ist die Brücke zwischen stabilem Core und wachsender Fähigkeitsschicht:

1. Core-Status prüfen
2. Governance prüfen
3. Task-Historie reflektieren
4. Risiken ableiten
5. Vorschläge für den User erzeugen
6. Review-Paket speichern

## CLI

```bash
python main.py nightly-review
python main.py nightly-review --limit 50
python main.py nightly-review --no-write
```

Die gespeicherten Review-Pakete liegen unter:

```text
proposals/nightly_reviews/
```

## Harte Regel

`auto_changes_made` muss immer `false` bleiben. Der Nightly Review darf nur Vorschläge erzeugen.
