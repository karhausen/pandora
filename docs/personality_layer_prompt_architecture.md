# MVP 28.1 – Personality Layer & Prompt Architecture

MVP 28.1 ergänzt Pandora um eine getrennte Kommunikations- und Prompt-Schicht.

## Ziel

Pandora soll nicht nur wissen, was sie ist und was sie kann, sondern auch konsistent steuern, **wie** sie antwortet. Die Personality-Schicht ist bewusst von der Cognitive Identity getrennt:

- Cognitive Identity: Was Pandora ist, kann und nicht behaupten darf.
- Personality Layer: Ton, Ausführlichkeit und Antwortregeln.
- Prompt Architecture: stabile Layer für Identity, Personality, Grenzen, Aufgabe, Output-Vertrag und Safety Gate.

## Neue Dateien

- `core/personality_layer.py`
- `core/personality_layer_regression.py`
- `config/system/personality.json`

## CLI

```bash
python main.py personality-status
python main.py personality-profile
python main.py personality-style-contract
python main.py prompt-package "Bitte plane den nächsten sicheren Schritt"
python main.py prompt-preview "Bitte plane den nächsten sicheren Schritt"
python main.py personality-regression-run
```

## API

- `GET /api/cognitive/personality/status`
- `GET /api/cognitive/personality/profile`
- `GET /api/cognitive/personality/style-contract`
- `GET /api/cognitive/prompt/package?query=...`
- `GET /api/cognitive/prompt/preview?query=...`
- `GET /api/cognitive/personality/regression`

## Sicherheitsgrenze

Die Schicht ist read-only. Sie erstellt Prompt-Pakete und Stil-Verträge, ruft aber kein LLM auf, führt keine Tools aus, schreibt keinen Speicher und umgeht keine Freigabe.
