# Pandora Roadmap -- Phase 5: Cognitive Live Console

## Vor Phase 5 -- Evolution Stabilization

Diese Schritte schließen die Evolution-Architektur sauber ab und
beseitigen die heute identifizierten Schwachstellen der
Capability-Entscheidung.

### MVP 29.7.1 -- Capability Gap Guardrail Fix

-   Verhindert falsche Tool-Auswahl (z. B. Calculator statt
    Tool-Proposal)
-   Spezialfähigkeiten dürfen nicht durch generische Tools verdeckt
    werden
-   Integrationstest für Primzahlen-Workflow

### MVP 29.7.2 -- Semantic Capability Decision Engine

-   LLM bewertet User-Aufgabe gegen:
    -   Tool Registry
    -   Skill Registry
    -   Knowledge Registry
    -   Capability Graph
    -   Genome
-   Python validiert die Entscheidung
-   Keine Keyword- oder Pattern-Erkennung mehr
-   Kein `_looks_like_*`
-   Keine Capability-spezifische Logik im Python-Code

### MVP 29.7.3 -- Cognitive Execution Trace

-   Live-Sidebar
-   Local LLM / Cloud LLM / Python
-   Memory / Knowledge / Tool / Evolution
-   Routing Trace
-   Provider & Modell
-   Timeline
-   Developer Mode

### MVP 29.7.4 -- End-to-End Smoke Test Suite

-   Prime Number Tool
-   Word Count Tool
-   Morsecode Tool
-   ISBN Validator
-   Excel Merge Tool
-   QR Code Generator
-   Vollständiger Chat → Proposal → Approval → Tool → Observation →
    Evolution Test

------------------------------------------------------------------------

# Phase 5 -- Cognitive Live Console

-   [ ] MVP 30.0 -- Cognitive Live Console
-   [ ] MVP 30.1 -- Execution Event Bus
-   [ ] MVP 30.2 -- Live Component Monitor
-   [ ] MVP 30.3 -- Execution Timeline
-   [ ] MVP 30.4 -- Cognitive Routing Trace
-   [ ] MVP 30.5 -- Provider & Model Monitor
-   [ ] MVP 30.6 -- Performance Analytics
-   [ ] MVP 30.7 -- Interactive Debug Console
-   [ ] MVP 30.8 -- End-to-End Test Runner
-   [ ] MVP 30.9 -- Cognitive Replay
-   [ ] MVP 31.0 -- Pandora Mission Control
