# MVP 27.1 – Adaptive Source Selection

Pandora nutzt ab MVP 27.1 eine adaptive Quellenauswahl vor dem eigentlichen Lesen von Kontext.

Wichtig: Das LLM empfiehlt nur Informationsräume. Python normalisiert, priorisiert und validiert diese Empfehlungen gegen Profil- und Governance-Regeln.

## Ablauf

```text
User Request
↓
Cognitive Planning Engine
↓
Adaptive Source Selection
↓
Python Policy Validation
↓
Selected Sources
↓
Context Builder
```

## Sicherheitsmodell

Die Adaptive Source Selection:

- liest keine Dateien,
- führt keine Tools aus,
- erzeugt keinen Code,
- schreibt nichts,
- aktiviert nichts.

## CLI

```bash
python main.py adaptive-source-selection-status
python main.py adaptive-source-select "Was war meine letzte Notiz?"
```

## API

```text
GET /api/cognitive/adaptive-source-selection/status
GET /api/cognitive/adaptive-source-selection/preview?query=...
```

## Ziel

Variierende Formulierungen sollen denselben sinnvollen Kontext-Raum treffen, ohne starre If-Else-Logik und ohne LLM-Direktzugriff auf Dateien.
