# Konfiguration

Pandora trennt Standardkonfiguration, lokale Konfiguration und Secrets.

## Wichtige Dateien

```text
config/llm/llm_config.json              Standard-LLM-Konfiguration
config/llm/llm_config.local.json        lokale/private Overrides, nicht im Release
config/llm/llm_config.local.example.json Beispiel
config/tools/tool_registry.json         Tool Registry
config/tools/execution_policy.json      Ausführungsregeln
config/skills/skill_registry.json       Skill Registry
```

## Profile

Typische Profile:

```text
private  lokale Modelle + private Cloud-LLMs
company  Company-LLM + interne Endpunkte
```

## LLM Routing

Aufgaben werden nach Zweck geroutet:

```text
chat
planning
tool_design
tool_code_generation
code_review
core_review
maintenance
night_mode
```

Bearbeitung in der GUI:

```text
/llm-profiles
```

Änderungen werden als lokale Konfiguration gespeichert und auditiert.

## Secrets

Secrets gehören nicht ins Repository und nicht ins Release-ZIP.

Erlaubt:

```text
.env
Umgebungsvariablen
config/llm/llm_config.local.json
```

Nicht erlaubt:

```text
API Keys in README.md
API Keys in docs/
API Keys in config/llm/llm_config.json
API Keys in user_knowledge/public/
```
