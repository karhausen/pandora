# MVP 30.3.1 – Proposal Confirmation Gate

## Ziel

Der Primzahlen-Test zeigte: Pandora durfte bei einer unklaren Tool-Anfrage noch in eine freie Chat-Antwort mit allgemeinen Tool-Entwicklungs-Schritten fallen.

## Änderungen

- `clarify` wird im `ChatService` jetzt als eigener Ausführungspfad behandelt.
- Tool-/Capability-Erstellung benötigt eine explizite Bestätigung für eine dauerhafte Pandora-Capability.
- Ein fehlendes Tool führt nicht mehr automatisch zu `tool_development`.
- Unklare Tool-Anfragen werden als Klärung beantwortet: einmalige Berechnung mit bestehenden Fähigkeiten oder dauerhaftes Proposal?
- Cognitive Reasoning Prompt geschärft: keine generischen Tool-Entwicklungs-Anleitungen als finaler Chat-Pfad.

## Regression

- 14 Tests bestanden.
