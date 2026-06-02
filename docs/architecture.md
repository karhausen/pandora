# Architektur

Pandora besteht aus Core, Tools, Skills, Memory, Agent Loop, Capability Expansion, Learning und Web-GUI.


## MVP 19.1 – Memory Recall Agent

Der Memory Recall Agent liegt zwischen Coordinator und normalem Chat-Fallback. Er prüft direkte Erinnerungsfragen gegen `ConversationMemory` und liefert bei Treffer eine strukturierte `MemoryRecallResult`-Antwort. Dadurch bleibt die Entscheidung nachvollziehbar und benötigt keine externe Datenbank oder Cloud-Komponente.

Aktueller Recall-Umfang:

- gespeicherter Name
- Fragen wie `Wie heiße ich?`
- indirekte Formulierungen wie `Ich habe meinen Namen vergessen.`
- Rückfragen wie `Weißt du noch, wie ich heiße?`

Der bestehende `ConversationMemory.answer_from_memory()` bleibt als Kompatibilitäts-Fassade erhalten und delegiert an den neuen Agenten.


## MVP 19.2 – Tool Development Agent

Der Tool Development Agent liegt im Coordinator vor dem Planner/Worker-Pfad. Er prüft, ob eine Aufgabe eine noch fehlende Tool-Fähigkeit beschreibt. Bei Treffer erzeugt er über den bestehenden `ToolProposalManager` einen Proposal-Ordner mit Code, Tests und Validierungsdaten.

Routing-Reihenfolge im Coordinator:

```text
Memory Recall
↓
Tool Development
↓
Planner / Worker
↓
Chat Fallback
```

Verantwortlichkeiten:

- `CapabilityDetector`: erkennt bekannte oder analysierte Capability-Lücken
- `ToolDevelopmentAgent`: entscheidet, ob Tool-Entwicklung zuständig ist
- `ToolProposalManager`: erzeugt Proposal, Code, Tests und Validation
- `ToolActivationManager`: bleibt für spätere manuelle Aktivierung zuständig

Sicherheitsprinzip:

Tool-Erzeugung bleibt ein Proposal-Prozess. Der Agent darf Vorschläge erzeugen, aber keine neuen Tools ungeprüft aktivieren.
