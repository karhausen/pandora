# MVP 21.1.1 – Release Packaging System

Pandora-Releases dürfen keine lokalen Laufzeitdaten, Caches oder Secrets enthalten. Dieses MVP führt eine reproduzierbare Packaging-Schicht ein.

## Ziele

- ZIPs vor der Weitergabe bereinigen
- Runtime-Artefakte entfernen
- lokale Konfigurationen und Secrets blockieren
- Manifest mit Datei-Hashes erzeugen
- Audit als Quality Gate nutzen

## Wichtige Dateien

- `scripts/release_audit.py` prüft einen Projektbaum auf Release-Blocker.
- `scripts/export_release.py` erzeugt ein bereinigtes ZIP.
- `.gitignore` verhindert lokale Runtime-Dateien im Repository.
- `.dockerignore` hält Docker-Builds sauber.
- `release_manifest.json` wird im Release-ZIP erzeugt.

## Export

```bash
python scripts/export_release.py --version mvp-21.1.1-release-packaging
```

Optional ohne Testlauf:

```bash
python scripts/export_release.py --version mvp-21.1.1-release-packaging --skip-tests
```

## Audit

```bash
python scripts/release_audit.py .
python scripts/release_audit.py . --json
```

## Blockierte Inhalte

- `.env`, `.env.*` außer `.env.example`
- `*.local.json` außer `*.local.example.json`
- `__pycache__`, `.pytest_cache`, `.venv`, `venv`, `dist`, `build`
- `*.pyc`, `*.log`, Coverage-Artefakte
- nicht-leere Runtime-Ordner wie `logs`, `sandbox/runs`, `sandbox/tmp`
- offensichtliche Inline-Secrets in Textdateien

## Sicherheitsregel

Das Release-Packaging ist kein Komfort-Feature, sondern ein Core-Sicherheitsbaustein. Besonders bei Company-Profilen darf Pandora niemals lokale Endpunkte, Tokens, Logs oder Memory-Dumps in ein Release-ZIP schreiben.
