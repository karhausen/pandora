# Pandora Agent MVP 3.0

Lokaler modularer Python-Agent mit stabilem Core und kontrollierter Tool-Erzeugung.

## Enthalten

- Core-CLI
- Tool Registry
- Tool Discovery
- gehärteter Tool Executor
- Tool Runtime SQLite DB
- Heartbeat
- Safe Mode
- Capability Analyzer
- Tool Generator
- Tool Validator
- Tool Tester
- Tool Lifecycle Manager
- Reflection Log
- Proposal-Verzeichnis für generierte Tools

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
python main.py tool-list
python main.py run-tool echo --input "Hallo Agent"
python main.py run-tool calculator --json '{\"expression\":\"2+3*4\"}'
python main.py memory
python main.py safe-mode
```

## Neue MVP-3-Befehle

Task analysieren:

```powershell
python main.py analyze "Bitte CSV Datei auswerten"
```

Fehlende Fähigkeit erkennen und Tool kontrolliert erzeugen:

```powershell
python main.py ensure-capability "Bitte CSV Datei auswerten" --auto-create
```

Danach:

```powershell
python main.py tools
python main.py tool-stats
python main.py reflections
```

CSV Tool testen:

```powershell
@"
name,value
a,1
b,2
c,3
"@ | Set-Content sample.csv

python main.py run-tool csv_reader --json "{\"path\":\"C:\\GitHub\\pandora\\sample.csv\"}"
```

## Sicherheitsregeln

Generierte Tools werden nicht direkt blind aktiviert.

Ablauf:

1. ToolSpec erzeugen
2. Proposal speichern
3. AST-Sicherheitsprüfung
4. Syntaxprüfung
5. Tool-Datei schreiben
6. Import-Test
7. Smoke-Test
8. Registrierung per Discovery

Blockiert werden u.a.:

- subprocess
- socket
- ctypes
- eval
- exec
- os.system
- shutil.rmtree
- open

Hinweis: Das MVP nutzt noch keine echte Prozess-Sandbox. Vor autonomer Tool-Erzeugung mit fremden LLM-Ausgaben muss eine härtere Sandbox folgen.

## Tests

```powershell
pytest
```

## Architekturregel

Der aktive Core wird nicht autonom überschrieben. MVP 3 erzeugt nur Tools und Proposals.
