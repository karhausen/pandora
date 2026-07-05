# MVP 29.7.3 – Cognitive Execution Trace

## Ziel

Pandora macht sichtbar, welche Hauptkomponenten während einer Chat-Ausführung beteiligt waren.
Die User-GUI erhält eine rechte Live-Sidebar mit LED-Status für Local LLM, Cloud LLM, Python,
Tool, Knowledge, Memory, Evolution und Proposal.

## Enthalten

- Neues Paket `core/execution_trace`
- `ExecutionTraceManager` mit persistentem Eventlog unter `data/execution_trace`
- API-Endpunkte:
  - `/api/execution-trace/status`
  - `/api/execution-trace/current`
  - `/api/execution-trace/events`
  - `/api/execution-trace/start`
  - `/api/execution-trace/record`
  - `/api/execution-trace/from-result`
  - `/api/execution-trace/reset`
- CLI-Kommandos:
  - `python main.py execution-trace status`
  - `python main.py execution-trace current`
  - `python main.py execution-trace events`
  - `python main.py execution-trace reset`
  - `python main.py execution-trace start --task "..."`
- User-GUI Sidebar `Pandora Live`
- LED-Style Komponentenstatus
- Kurz-Timeline der letzten Trace-Events
- Integration in CLI/API Selftests

## Sicherheitsprinzip

Execution Trace ist rein beobachtend. Der Trace entscheidet nichts, aktiviert nichts und verändert keine
Tools, Proposals oder Core-Dateien.
