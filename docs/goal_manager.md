# MVP 27.3 – Goal Manager

Der Goal Manager leitet aus einer Anfrage, dem Cognitive Plan und der Central Decision Engine **Zielkandidaten** ab.

Er ist bewusst konservativ:

- keine automatische Speicherung
- keine Tool-Ausführung
- keine Tool-Aktivierung
- keine Core-Änderung
- keine Knowledge-/Vault-Schreiboperation

## Aufgabe

Der Goal Manager beantwortet intern:

- Gehört die Anfrage zu einem längerfristigen Ziel?
- Betrifft sie Tools, Knowledge, Core oder Planung?
- Welcher Zielvorschlag sollte reviewt werden?
- Wie hoch ist die Priorität?

## CLI

```bash
python main.py goal-manager-status
python main.py goal-propose "Ich brauche ein Tool fuer Aktienkurse"
```

## API

- `GET /api/cognitive/goal-manager/status`
- `GET /api/cognitive/goal-manager/preview?query=...`

## Sicherheitsregel

Jeder Zielkandidat bleibt ein Vorschlag. Persistenz, Umsetzung oder Core-Änderungen benötigen später explizite Review- und Freigabeschritte.
