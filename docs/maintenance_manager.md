# MVP 21.2 – Maintenance Manager

Der Maintenance Manager ist Pandoras kontrollierter Wartungsmodus. Er ist die technische Vorstufe zur langfristigen Variante C: Day Mode / Night Mode.

## Ziel

Pandora soll außerhalb der normalen Nutzung prüfen, ordnen und Vorschläge erzeugen, aber keine gefährlichen Änderungen selbst aktivieren.

Der Manager darf:

- Nightly Governance Review starten
- Release Audit ausführen
- Wartungsreport schreiben
- erwartete Runtime-Ordner und `.gitkeep`-Marker anlegen
- Ergebnisse im Memory Gateway protokollieren

Der Manager darf nicht:

- Core-Dateien ändern
- Tools installieren oder aktivieren
- Skills aktivieren
- Pakete installieren
- Netzwerkzugriffe ausführen
- Secrets oder Profile ändern

## CLI

Status anzeigen:

```bash
python main.py maintenance-status
```

Geplanten Lauf prüfen, ohne Dateien zu schreiben:

```bash
python main.py maintenance-run --dry-run --force
```

Einmaligen Wartungslauf starten:

```bash
python main.py maintenance-run --force --limit 200
```

Ohne `--force` läuft der Manager nur im konfigurierten Wartungsfenster:

```bash
python main.py maintenance-run --window-start 02:00 --window-end 05:00
```

## Empfohlene Auslösung

Kurzfristig robust per Betriebssystem:

Linux/macOS Cron:

```cron
0 3 * * * cd /path/to/pandora && .venv/bin/python main.py maintenance-run --window-start 02:00 --window-end 05:00
```

Windows Task Scheduler:

```text
03:00 Uhr täglich
python main.py maintenance-run --window-start 02:00 --window-end 05:00
```

Docker:

```bash
docker compose run --rm pandora python main.py maintenance-run --force
```

## Architekturentscheidung

Der Heartbeat bleibt für Überleben, Gesundheit und Recovery zuständig. Der Maintenance Manager übernimmt Wartung, Reports und spätere Night-Mode-Aufgaben. So bleibt der Core übersichtlich und der Heartbeat wird nicht überladen.
