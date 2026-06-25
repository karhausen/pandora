# Tool Design Agent

MVP 19.6 führt einen echten Design-Schritt vor der Code-Erzeugung ein.

## Ziel

Aus einer fehlenden Fähigkeit wie `weather_lookup` wird zunächst ein prüfbarer Tool-Vertrag:

```json
{
  "capability": "weather_lookup",
  "tool_id": "weather_lookup",
  "input_schema": {"location": "str"},
  "output_schema": {"temperature": "float", "condition": "str"},
  "security_level": "LIMITED",
  "requires_network": true,
  "dependencies": [],
  "test_cases": []
}
```

## Rollen

```text
Tool Development Agent  -> erkennt fehlende Fähigkeit und startet Entwicklung
Tool Design Agent       -> erstellt Design/Vertrag
Tool Proposal Manager   -> erzeugt Proposal-Dateien
Tool Validator          -> prüft Code und Tests
Tool Activation Manager -> aktiviert nur nach manueller Freigabe
```

## Model Routing

`tool_design` routet standardmäßig auf `cloud_expert`, damit echte Tool-Verträge später von einem stärkeren Modell erstellt werden können.

Für Tests kann `--provider mock` genutzt werden.

## Dateien im Proposal

```text
tool_proposals/<ID>/
├─ tool_design.json
├─ proposal.json
├─ validation.json
├─ generated_tools/<tool_id>.py
└─ tests/test_<tool_id>.py
```

## Sicherheitsregeln

- Netzwerkzugriff -> mindestens `LIMITED`
- Shell-Zugriff nur mit sehr guter Begründung
- Secrets nur über ENV/config, niemals im Code
- keine automatische Aktivierung
