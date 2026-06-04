# Tool Review & Policy-Aware Validation

MVP 19.8 ergänzt einen lokalen `ToolReviewAgent`. Cloud-Modelle dürfen Tool-Code erzeugen, aber Pandora validiert diesen Code lokal anhand des ToolDesigns und einer Sicherheits-Policy.

## Grundprinzip

```text
ToolDesign
↓
CloudToolCodeGenerator
↓
ToolReviewAgent
↓
pytest
↓
Proposal Status
```

## Policy-Regeln

SAFE-Tools:

- kein Netzwerk
- keine Dateioperationen
- keine Shell
- keine gefährlichen Imports

LIMITED-Tools mit `requires_network=true`:

- erlaubt: `urllib.request`, `urllib.parse`, `urllib.error`, `json`, `os`
- Netzwerkaufrufe müssen `timeout=` setzen
- API-Keys und Konfiguration müssen über Environment Variablen kommen
- verboten bleiben: `requests`, `httpx`, `socket`, `subprocess`, `eval`, `exec`, `open`, `ctypes`, `multiprocessing`

## CLI

```bash
python main.py tool-review-file path/to/tool.py --design path/to/tool_design.json
```

## API

```text
POST /tool-review/review
```

Body:

```json
{
  "code": "...",
  "design": {"requires_network": true, "security_level": "LIMITED"}
}
```

## Warum lokal?

Der Cloud Expert darf Code vorschlagen. Die Entscheidung, ob Code sicher genug für Pandora ist, bleibt lokal und nachvollziehbar.
