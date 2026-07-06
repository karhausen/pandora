# MVP 29.4.3 – Tool Generator Prime Contract Fix

Fixes the missing `_looks_like_prime_tool` helper in both `ToolGenerator` and `ToolTestGenerator`.

Important: this helper is only a generator-contract fallback. Capability-gap routing remains LLM-based and must not rely on keyword/pattern detection.

Validation target:

```bash
python main.py selftest integration
```

Manual chat tests:

- `Ich möchte Wörter zählen können.`
- `Ich brauche ein Tool, das Prim-Zahlen berechnet.`

Expected result: both create validated tool proposals instead of failing with AttributeError.
