# MVP 25.4 – Capability Analyzer

Der Capability Analyzer ist die Python-seitige Diagnose-Schicht nach dem Request Interpreter.

## Aufgabe

Er bewertet strukturierte Interpreter-Empfehlungen und prüft gegen Pandoras realen Systemzustand:

- vorhandene Tools
- vorhandene Skills
- empfohlene Quellenräume
- gemeldete Capability-Gaps
- explizite Tool-/Core-/Knowledge-Wünsche

## Nicht-Aufgabe

Der Analyzer führt nichts aus.

Er darf nicht:

- Tools ausführen
- Python-Code erzeugen
- Tools aktivieren
- Dateien lesen
- Core-Änderungen durchführen
- Freigaben ersetzen

## Pipeline

```text
User Request
↓
Request Interpreter
↓
Capability Analyzer
↓
Python Orchestrator
↓
Context Builder / Tool Factory / Review Workflow
```

## Ergebnis

Der Analyzer erzeugt eine strukturierte Diagnose:

- `gaps`
- `gap_summary`
- `recommended_actions`
- `priority`
- `confidence`
- `safety`

## Grundregel

```text
LLM empfiehlt Fähigkeiten.
Python prüft reale Verfügbarkeit.
Änderungen laufen immer über Review, Tests, Governance und User-Freigabe.
```
