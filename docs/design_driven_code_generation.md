# MVP 20.5 – Design Driven Code Generation & Placeholder Detection

Pandora erzeugt Tool-Code ab MVP 20.5 nicht mehr nur aus einer Capability, sondern strikt aus dem `ToolDesign`.

## Regeln

- `TOOL_META.output_schema` muss dem Design entsprechen.
- `run(payload)` muss alle Keys aus `output_schema` zurückgeben.
- Rückgabewerte müssen zum Schema passen.
- Generischer Echo-/Dummy-Code ist verboten, wenn das Schema andere Felder verlangt.
- Cloud-Fehler erzeugen keinen scheinbar gültigen Dummy-Code.
- `generate_with_llm` akzeptiert Proposals nur, wenn Static Review, pytest und Semantic Quality Gate erfolgreich sind.

## Blockierte Placeholder

Beispiele:

```python
return {"text": str(text)}
return payload
pass
raise NotImplementedError
# TODO
```

## Beispiel

Wenn das Design verlangt:

```json
{
  "output_schema": {
    "ticker": "string",
    "price": "number"
  }
}
```

Dann ist dieser Code ungültig:

```python
return {"text": str(text)}
```

Das Proposal bleibt `FAILED`.
