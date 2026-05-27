# Pandora Agent MVP 9A.0

MVP 9A integriert die erste echte LLM-Schicht.

Neu:
- LLM Runtime
- Provider-Abstraktion
- Mock Provider für stabile Offline-Tests
- Ollama Provider
- OpenAI Provider
- Modellrouting nach Task-Typ
- Prompt Manager
- Prompt-Verzeichnis
- strukturierte JSON-Ausgaben
- Pydantic-Validierung
- LLM-gestützte Task-Analyse
- LLM-Governance-Grundregeln

Wichtig: Das LLM führt weiterhin nichts direkt aus. Es erzeugt Vorschläge und strukturierte Analysen. Der Core entscheidet.

## Befehle

```powershell
python main.py status
python main.py heartbeat
python main.py tools
python main.py llm-config
python main.py llm-analyze "Bitte rechne 2+3*4"
python main.py llm-complete "Hallo Pandora"
```

## Ollama

```powershell
python main.py llm-analyze "Plane diese Aufgabe" --provider ollama --model llama3.1:8b
```

## OpenAI

```powershell
$env:OPENAI_API_KEY="..."
python main.py llm-analyze "Plane diese Aufgabe" --provider openai --model gpt-4.1-mini
```

## API

```powershell
python main.py api
```

Neue Endpunkte:
- GET /llm/config
- POST /llm/analyze
- POST /llm/complete

## Tests

```powershell
pytest
```

## Nächster Schritt

MVP 9B: Controlled Self-Improvement mit Patch-Proposals, Diff Manager, Code Review, Regression Runner und Approval Pipeline.
