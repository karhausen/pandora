# MVP 27.5 – Weekly/Monthly Review Cycles

Der Review Cycle Engine bündelt Ziele, Prioritäten und zentrale Entscheidungen in einem überprüfbaren Wochen- oder Monatsreview.

## Zweck

Pandora soll regelmäßig erkennen:

- welche Ziele wichtig sind,
- welche Tool-/Knowledge-/Core-Gaps offen sind,
- welche Punkte Freigabe brauchen,
- was als Nächstes vorbereitet werden sollte.

## Sicherheitsregel

Der Review Cycle Engine führt nichts aus.

Er darf nicht:

- Tools aktivieren,
- Core-Code ändern,
- Obsidian oder Knowledge schreiben,
- Memory dauerhaft verändern,
- Releases erzeugen.

Er erzeugt nur Review-Pakete und Freigabefragen.

## CLI

```bash
python main.py review-cycle-status
python main.py review-cycle-preview "Pandora soll sich regelmäßig verbessern" --cadence weekly
python main.py review-cycle-preview "Pandora soll sich strategisch weiterentwickeln" --cadence monthly
```

## API

```text
GET /api/cognitive/review-cycle/status
GET /api/cognitive/review-cycle/preview?query=...&cadence=weekly
```
