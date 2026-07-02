# MVP 28.2 – User GUI Simplification

## Ziel

Die User-GUI soll nicht mehr wie ein Admin-Dashboard wirken. Sie ist der einfache Chat-Einstieg für den Nutzer. Alles, was Wartung, Konfiguration, Review, Approval, Knowledge-Verwaltung oder Diagnose ist, liegt hinter einem einzigen Einstieg: `/maintenance`.

## Ergebnis

- `/` ist Chat-first.
- `/maintenance` ist der strukturierte Wartungsbereich.
- Die Startseite zeigt keine Kartenflut mehr.
- Routing-Information bleibt sichtbar, aber nicht als Konfigurationsoberfläche.
- Technische Details bleiben einklappbar.

## Sicherheitsgrenzen

Die neue Schicht ist read-only. Sie beschreibt und rendert Navigation. Sie führt keine Tools aus, gibt keine Proposals frei, installiert nichts und schreibt keine Konfiguration.

## API

`GET /api/gui/user-simplification/status` liefert den Navigationsvertrag für User-GUI und Maintenance-Bereich.
