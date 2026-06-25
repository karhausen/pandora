# MVP 23.3.3 – Registration Validation

Pandora prüft vor Releases nun, ob CLI-, API- und GUI-Registrierungen zusammenpassen.

## Warum?

In MVP 23.3.1 konnte `main.py api` nicht starten, weil der ArgumentParser einen nicht vorhandenen CLI-Handler referenziert hat. Diese Prüfung soll solche Integrationsfehler früh erkennen.

## CLI

```bash
python main.py registration-validate
python main.py registration-validate --strict
python main.py registration-validate-cli
python main.py registration-validate-api
python main.py registration-validate-gui
```

## Prüfungen

- CLI: Jeder `set_defaults(func=cmd_...)` Handler muss existieren.
- API: `core.api` muss importierbar sein und Routen müssen Endpoints haben.
- GUI: `fetch('/api/...')` Aufrufe werden gegen bekannte API-Routen geprüft.

## Release-Regel

Für echte Releases sollte mindestens laufen:

```bash
python main.py --help
python main.py registration-validate --strict
python main.py api --help
```

Die Prüfung ruft keine Live-LLMs auf und führt keine Tools aus.

## API

```text
GET /api/system/registration-validation
GET /api/system/registration-validation/cli
```

Diese Endpunkte sind für Operations/Diagnostics gedacht und verändern keinen Zustand.
