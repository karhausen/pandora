# MVP 26.3 – Proposal Review Loop

Der Proposal Review Loop ist die kontrollierte Review-Schicht nach der Benutzerfreigabe zur Vorschlagserstellung.

Er beantwortet nicht selbst, baut keinen Code und ändert nichts am System. Er führt nur durch den Review-Zustand:

1. Pandora erkennt einen Vorschlagsbedarf.
2. Der Benutzer erlaubt die Ausarbeitung.
3. Ein Workflow erzeugt einen Proposal-Payload.
4. Der Benutzer reviewed: `passt`, `nachbessern` oder `ablehnen`.
5. Erst bei `passt` darf der Vorschlag in den nächsten kontrollierten Workflow.

## Garantien

- keine Codegenerierung
- keine Tool-Aktivierung
- keine Core-Änderung
- keine Knowledge-Schreiboperation
- Freigabe bleibt zwingend

## CLI

```bash
python main.py proposal-review-loop-status
python main.py proposal-review-loop-preview "Baue ein Tool für historische Aktienkurse"
python main.py proposal-review-loop-preview "Baue ein Tool für historische Aktienkurse" --payload-json '{"purpose":"Demo"}' --review-decision passt
```

## API

- `GET /api/cognitive/proposal-review-loop/status`
- `GET /api/cognitive/proposal-review-loop/preview?query=...`
