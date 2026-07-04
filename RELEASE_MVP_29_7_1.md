# MVP 29.7.1 – Capability Gap Guardrail Fix

## Ziel

Behebt den End-to-End-Fehler aus dem Primzahlen-Test:

> Ich brauche ein Tool, das Prim-Zahlen berechnet.

Pandora darf daraus nicht mehr schließen:

> Es fehlt kein neues Tool oder die Fähigkeit ist bereits vorhanden.

## Änderung

Der LLM Capability Gap Analyzer wurde gehärtet:

- Wenn das LLM eine gewünschte Capability nennt, aber kein valides vorhandenes Tool liefert, wird dies als Capability Gap behandelt.
- Wenn das LLM ein vorhandenes Tool vorschlägt, validiert Python generisch anhand der Tool-Metadaten, ob dieses Tool die gewünschte Capability tatsächlich beschreibt.
- Breite Tools wie `calculator` dürfen nicht mehr automatisch als passend für spezialisierte Fähigkeiten wie `prime_number_calculation` akzeptiert werden.
- Die Prüfung bleibt generisch: keine `_looks_like_*`-Methoden, keine capability-spezifischen Sonderfälle.

## Neuer Selftest

`python main.py selftest integration` prüft jetzt zusätzlich:

- inkonsistente LLM-Entscheidung: Capability vorhanden, aber `tool_needed=false`
- over-broad Match: LLM schlägt `calculator` für `prime_number_calculation` vor

Beide Fälle müssen als Gap erkannt werden.
