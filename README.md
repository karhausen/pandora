# Pandora Agent MVP 4.0

Lokaler modularer Python-Agent mit stabilem Core, kontrollierter Tool-Erzeugung und erstem Skill-System.

## Was ist neu in MVP 4?

MVP 4 ergänzt Skills als wiederverwendbare Workflows über mehrere Tools.

Neu:

- `SkillRegistry`
- `SkillExecutor`
- `SkillManager`
- Skill-Discovery aus `/skills/*.json`
- Skill-Proposals unter `/proposals/skills`
- Skill-Runtime-Logging in SQLite
- CLI:
  - `skills`
  - `skill-list`
  - `run-skill`
  - `create-demo-skill`
  - `skill-runs`
- `--file` für stabile JSON-Payloads ohne PowerShell-Quoting-Stress

## Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py status
```

## Basisbefehle

```powershell
python main.py heartbeat
python main.py tools
python main.py skills
python main.py tool-list
python main.py skill-list
python main.py memory
python main.py safe-mode
```

## Tool ausführen

```powershell
python main.py run-tool echo --input "Hallo Agent"
python main.py run-tool calculator --json '{\"expression\":\"2+3*4\"}'
```

Stabiler mit Datei:

`payload_calc.json`

```json
{
  "expression": "2+3*4"
}
```

```powershell
python main.py run-tool calculator --file payload_calc.json
```

## Skill ausführen

Vorinstallierter Demo-Skill:

```powershell
python main.py run-skill echo_then_upper --json '{\"text\":\"Hallo Agent\"}'
```

Stabiler mit Datei:

`payload_skill.json`

```json
{
  "text": "Hallo Agent"
}
```

```powershell
python main.py run-skill echo_then_upper --file payload_skill.json
```

Erwartetes Ergebnis:

```json
{
  "upper": {
    "text": "HALLO AGENT"
  }
}
```

## Skill-Aufbau

Skills liegen als JSON-Dateien unter:

```text
skills/
```

Beispiel:

```json
{
  "id": "echo_then_upper",
  "name": "Echo Then Upper",
  "description": "Echoes input text and converts it to uppercase.",
  "required_tools": ["echo", "uppercase"],
  "steps": [
    {
      "id": "echo",
      "type": "tool",
      "tool_id": "echo",
      "input_map": {
        "text": "input.text"
      },
      "save_as": "echo"
    },
    {
      "id": "upper",
      "type": "tool",
      "tool_id": "uppercase",
      "input_map": {
        "text": "context.echo.text"
      },
      "save_as": "upper"
    }
  ]
}
```

## Tests

```powershell
pytest
```

## Architekturregel

Auch MVP 4 verändert den aktiven Core nicht autonom.

Autonom oder halbautonom erweiterbar sind weiterhin nur:

- Tools
- Skills
- Proposals
- Workflows

Geschützt bleiben:

- Heartbeat
- Rollback
- Recovery
- Security
- Config
- aktiver Core

## Nächster Schritt: MVP 5

MVP 5 sollte Reflection und Evolution verbessern:

- aus erfolgreichen Tool-Ketten Skill-Vorschläge erzeugen
- wiederkehrende Muster erkennen
- schlechte Tools markieren
- Skill-Qualität bewerten
- Verbesserungsvorschläge speichern
- noch keine direkte Core-Selbstmodifikation
