# Pandora Agent MVP 1.5

Kleiner, stabiler Python-Core für einen lokalen autonomen Agenten. Fokus: Core zuerst, keine unkontrollierte Selbstüberschreibung.

## Enthalten

- stabile Paketstruktur mit `__init__.py`
- CLI mit `task`, `status`, `heartbeat`, `tools`, `tool-list`, `run-tool`, `memory`, `safe-mode`
- Tool Registry mit persistenten Metadaten in `tools/registry.json`
- Tool Executor mit Timeout, Fehlerbehandlung und Laufzeitmessung
- Beispiel-Tools: `calculator`, `echo`
- Memory: Short-Term JSON, Long/Episodic/Semantic SQLite
- Heartbeat für Core-Komponenten
- Safe-Mode-Entscheidung
- pytest-Tests

## Installation

```powershell
cd C:\GitHub\pandora
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Built-in Tools registrieren

```powershell
python tools\register_builtin_tools.py
```

Das Script setzt den Projektpfad selbst. `PYTHONPATH` ist nicht mehr nötig.

## Startbefehle

```powershell
python main.py status
python main.py heartbeat
python main.py tools
python main.py tool-list
python main.py run-tool echo --input "Hallo Agent"
python main.py run-tool calculator --json '{"expression":"2+3*4"}'
python main.py task "berechne 10 / 2"
python main.py memory
python main.py safe-mode
```

Hinweis für PowerShell: Falls JSON mit einfachen Anführungszeichen Probleme macht, nutze:

```powershell
python main.py run-tool calculator --json "{`"expression`":`"2+3*4`"}"
```

## Tests

```powershell
python -m pytest
```

Erwartung: `4 passed`.

## Architektur

```text
agent/
├── main.py
├── core/
│   ├── agent_core.py
│   ├── config.py
│   ├── heartbeat.py
│   ├── llm_client.py
│   ├── memory.py
│   ├── planner.py
│   ├── recovery.py
│   ├── reflection.py
│   ├── security.py
│   ├── tool_executor.py
│   └── tool_registry.py
├── tools/
│   ├── calculator.py
│   ├── echo.py
│   ├── register_builtin_tools.py
│   └── registry.json
├── skills/
├── memory/
├── logs/
└── tests/
```

## Nächster Schritt

MVP 2: Tool-System härten.

- Tool-Test-Command ergänzen
- Tool-Validierung ausbauen
- Sicherheitsstufen erzwingen
- Tool-Run-Historie zusätzlich in SQLite speichern
- schlechte Tools automatisch markieren, aber nicht löschen
