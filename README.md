# Pandora Agent MVP 9B.0

MVP 9B.0 ergänzt einen generischen OpenAI-kompatiblen Provider.

Damit kann Pandora lokal mit LM Studio, vLLM, LocalAI, OpenRouter, LiteLLM oder anderen OpenAI-kompatiblen Servern sprechen.

## Standard für Thomas

`memory/llm_config.json` ist vorbereitet für LM Studio:

```json
{
  "local_fast": {
    "type": "openai_compatible",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "default_model": "qwen/qwen3-1.7b"
  }
}
```

Standardrouting:

- chat → local_fast
- planning → local_fast
- tool_selection → local_fast
- reflection → local_fast
- tool_generation → openai, fallback mock
- core_review → openai, fallback mock

Wenn LM Studio nicht läuft, fällt Pandora automatisch auf `mock` zurück.

## LM Studio starten

In LM Studio:

```text
Developer → Local Server → Start Server
```

Dann Modell laden:

```text
qwen/qwen3-1.7b
```

Server:

```text
http://localhost:1234/v1
```

## Befehle

```powershell
python main.py status
python main.py heartbeat
python main.py llm-config
python main.py llm-analyze "Bitte rechne 2+3*4"
```

Explizit LM Studio:

```powershell
python main.py llm-analyze "Bitte analysiere diese Aufgabe" --provider local_fast
```

Explizit Mock:

```powershell
python main.py llm-analyze "Bitte rechne 2+3*4" --provider mock
```

Freier Call:

```powershell
python main.py llm-complete "Hallo Pandora" --provider local_fast
```

## API

```powershell
python main.py api
```

Endpunkte:

```text
GET  /llm/config
POST /llm/analyze
POST /llm/complete
```

## Tests

```powershell
pytest
```

## Wichtig

Das LLM führt weiterhin nichts direkt aus.

LLM:
- analysiert
- schlägt Tools vor
- schlägt Skills vor
- erzeugt strukturierte JSON-Ausgaben

Core:
- validiert
- entscheidet
- führt aus
- schützt Heartbeat, Rollback, Recovery und Security


## Fallback-Verhalten

Wenn `local_fast` / LM Studio nicht erreichbar ist, fällt Pandora automatisch auf `mock` zurück. Der lokale Provider hat einen kurzen Timeout, damit die CLI nicht lange blockiert.


## Timeout-Konfiguration

LM Studio braucht beim ersten Request oft länger als 2 Sekunden. Deshalb ist `local_fast` jetzt auf 20 Sekunden gesetzt:

```json
"local_fast": {
  "timeout": 20.0,
  "connect_timeout": 3.0,
  "read_timeout": 20.0
}
```

CLI-Override:

```powershell
python main.py llm-analyze "Bitte rechne 2+3*4" --provider local_fast --timeout 30
```

Für schnelle Offline-Tests:

```powershell
python main.py llm-analyze "Bitte rechne 2+3*4" --provider mock
```


# MVP 9B.0 – Controlled Self-Improvement

Neu:
- Patch Proposal Store
- Diff Manager
- Code Review
- Regression Runner
- Approval Manager
- Improvement Manager
- Sandbox-Validation
- Approved Snapshot Preparation

Wichtig: Auch MVP 9B überschreibt den aktiven Core nicht direkt.

## Beispiel

```powershell
python main.py propose-readme-improvement --title "Demo" --note "Kontrollierte Verbesserung."
python main.py improvement-list
python main.py improvement-show <ID>
python main.py improvement-validate <ID>
python main.py improvement-approve <ID>
python main.py improvement-prepare-snapshot <ID>
```
