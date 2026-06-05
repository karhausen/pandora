# Generic Capability Gap Detection – MVP 20.4

MVP 20.4 ergänzt die deterministische Sicherheits-Fallback-Erkennung im Capability Gate.

## Ziel

Wenn das lokale LLM eine Live-Daten- oder explizite Tool-Anfrage falsch als Chat/Planner-Aufgabe bewertet, erkennt Pandora dennoch eine fehlende Capability.

## Beispiele

- `Ich brauche ein Tool um Aktienkurse abzurufen` → `stock_price_lookup`
- `Wie ist der aktuelle Dollar-Kurs?` → `exchange_rate_lookup`
- `Welche Primzahlen liegen zwischen 10 und 30?` → keine Tool-Lücke

## Ablauf

```text
User Task
↓
LLM Capability Gate
↓
falls LLM fehlschlägt oder direkt antworten will
↓
Generic Capability Fallback
↓
Tool vorhanden?
  ├─ ja: Planner nutzt Tool
  └─ nein: Tool Development Proposal
```

Die Erkennung bleibt bewusst ein Fallback. Die LLM-Entscheidung ist weiterhin die erste Instanz.
