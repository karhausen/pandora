# MVP 29.6 – Decision Learning

## Ziel

Pandora speichert Benutzerentscheidungen zu Evolution-Proposals und leitet daraus eine kontrollierte Erfahrungsschicht ab.

## Enthalten

- `core/decision_learning/`
- Decision History Storage
- Decision Statistics
- Decision Pattern Detection
- Advisory Influence Signal für spätere Priorisierung
- Integration in Unified Proposal Queue: Queue-Entscheidungen werden automatisch mitgeschrieben
- CLI-Kommandos `learning ...` und `decision-learning ...`
- API-Endpunkte `/api/learning/...`
- Maintenance-Seite `/decision-learning`

## Sicherheitsregel

Decision Learning ist rein beratend:

- keine automatische Aktivierung
- keine automatische Ablehnung
- keine automatische Core-/Tool-Änderung
- Benutzerentscheidung bleibt Pflicht
