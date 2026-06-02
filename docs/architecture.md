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
