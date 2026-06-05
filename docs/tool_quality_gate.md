# Tool Quality Gate

MVP 20.3 ergänzt eine semantische Validierung für generierte Tools.

Die bisherigen Prüfungen bleiben erhalten:

```text
Static Review
↓
pytest
```

Neu ist zusätzlich:

```text
Semantic Validation
```

Der Quality Gate prüft, ob Code, `TOOL_META`, `ToolDesign` und Testfälle zusammenpassen.

## Was geprüft wird

- Tool-Modul ist importierbar.
- `TOOL_META` ist vorhanden.
- `run(payload)` ist vorhanden.
- `TOOL_META.output_schema` enthält die erwarteten Output-Keys.
- Rückgabewert von `run()` ist ein Dictionary.
- Rückgabe enthält alle erwarteten Output-Keys.
- Rückgabetypen passen zum Output-Schema.
- Design-Testfälle liefern die erwarteten Werte.

Beispiel:

```json
"output_schema": {
  "count": "integer"
}
```

Ein Tool, das nur folgendes zurückgibt, wird abgelehnt:

```python
return {"text": text}
```

weil `count` fehlt.

## CLI

```bash
python3 main.py tool-quality-proposal <PROPOSAL_ID>
```

## API

```text
GET /tool-quality/{proposal_id}
```

## Ziel

Ein Tool darf nicht nur syntaktisch korrekt sein und selbstgenerierte Tests bestehen. Es muss auch den Vertrag aus dem Tool Design erfüllen.
