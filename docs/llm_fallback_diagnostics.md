# MVP 22.9.2 – LLM Fallback Diagnostics

## Ziel

Wenn eine konfigurierte Chat-Route auf ein reales LLM zeigt, die Ausführung aber auf einen Fallback wie `mock` wechselt, muss Pandora das sichtbar machen.

## Verhalten

Die Chat-Ausführung enthält jetzt zusätzliche Felder:

```json
{
  "provider_name": "mock",
  "model": "mock-smart",
  "fallback_used": true,
  "primary_provider_name": "company_llm",
  "primary_model": "company-default-model",
  "fallback_reason": "Primary provider failed: ...",
  "routing_diagnostics": {
    "decision": "fallback",
    "primary_error": "...",
    "fallback_provider_name": "mock"
  }
}
```

## Wichtig

- Der Coordinator zeigt weiterhin die geplante/aufgelöste Route.
- Die Ausführung zeigt jetzt die tatsächlich verwendete Route.
- Ein Fallback auf `mock` gibt keine rohe Prompt-Echo-Antwort mehr aus, sondern eine freundliche Mock-Antwort.
- Die Fehlerursache des Primary Providers bleibt im Ausführungsbericht sichtbar.
