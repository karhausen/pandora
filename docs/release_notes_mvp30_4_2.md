# MVP 30.4.2 – Route Planner Message Order Fix

## Ziel

Der LLM-geführte Route-Planner bleibt erhalten. Der Router entscheidet weiterhin nicht fachlich.

## Fix

Einige OpenAI-kompatible Provider / hosted-vLLM-Setups lehnen mehrere `system`-Nachrichten ab oder akzeptieren System-Nachrichten nur ganz am Anfang. Der Route-Planner erzeugte durch Runtime-Kontext eine zweite System-Nachricht, wodurch der Provider mit HTTP 400 abbrach:

`System message must be at the beginning.`

## Änderung

- Genau eine `system`-Nachricht am Anfang.
- Runtime-Kontext wird in die `user`-Nachricht integriert.
- Keine Änderung an der Router-Architektur.
- Keine Keyword-Entscheidung.
- Keine Tools, Skills, Capability-Gap oder Tool Factory aktiviert.

## Tests

`25 passed`
