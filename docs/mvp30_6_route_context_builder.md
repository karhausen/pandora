# MVP 30.6 – Route Context Builder

Ziel dieses MVPs: Der LLM-geführte Route-Loop bleibt unverändert, aber der finale Kontext für die Antwort wird sauber, nachvollziehbar und begrenzt zusammengebaut.

## Regeln

- Der Router trifft weiterhin keine fachliche Entscheidung.
- Routen werden weiterhin vom LLM angefordert und von Python nur validiert/dispatcht.
- Tools, Skills, Capability Gap und Tool Factory bleiben deaktiviert.
- Der Context Builder entscheidet nicht, welche Quelle gebraucht wird.
- Der Context Builder bündelt nur bereits ausgeführte Routen-Ergebnisse.

## Neu

- `core/route_context_builder.py`
- Normalisierte Kontext-Metadaten:
  - Quellenliste
  - Quellanzahl
  - genutzte Zeichen
  - Kürzungsstatus
  - verwendete Routenarten
- Duplikate bei Quellen werden entfernt.
- Kontext wird über `PANDORA_ROUTE_CONTEXT_MAX_CHARS` begrenzt.

## Ablauf

```text
User
↓
Prompt Builder
↓
LLM wählt Route
↓
Router führt Route aus
↓
Route Loop sammelt Ergebnisse
↓
Route Context Builder baut finalen Kontext
↓
LLM formuliert Antwort
↓
User
```

## Tests

```text
32 passed
```
