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


## MVP 20.4.1 – Implicit Live Data Gap Detection

Die Erkennung wurde erweitert, damit Pandora nicht nur explizite Tool-Wünsche erkennt, sondern auch implizite Live-Datenfragen.

Beispiele:

```text
Wie wird das Wetter?
→ weather_lookup

Wie ist der Dollar-Kurs?
→ exchange_rate_lookup

Wie steht die BASF-Aktie?
→ stock_price_lookup
```

Wenn das lokale LLM freundlich direkt antworten möchte, prüft der Fallback trotzdem, ob die Aufgabe Live-Daten benötigt. Dadurch sollen Chat-Antworten wie „Ich habe keine aktuellen Daten“ nicht mehr echte Capability-Gaps verdecken.

## No-Dummy-Code Policy

Das Quality Gate blockiert generischen Template-Code, wenn er nicht zum `output_schema` passt. Beispiel: Ein Tool mit `output_schema` `ticker`, `price`, `change`, `change_percent` darf nicht nur `{"text": "..."}` zurückgeben.
