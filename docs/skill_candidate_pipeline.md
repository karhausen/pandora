# MVP 21.3 – Skill Candidate Pipeline

Pandora soll Fähigkeiten kontrolliert aufbauen. Die Skill Candidate Pipeline ist der erste konkrete Schritt in diese Richtung: Sie liest das Task Journal, erkennt wiederkehrende Tool-Muster und erzeugt daraus prüfbare Skill-Vorschläge.

## Grundregel

Die Pipeline ist **observe-only**:

- sie liest nur vorhandene Historie,
- sie erzeugt höchstens Proposal-Dateien,
- sie aktiviert keine Skills,
- sie ändert keine Registry,
- sie führt keinen generierten Skill aus,
- sie verändert keinen Core-Code.

## CLI

Status anzeigen:

```bash
python main.py skill-candidate-status
```

Trockenlauf:

```bash
python main.py skill-candidate-run --dry-run --force
```

Skill-Kandidaten erzeugen:

```bash
python main.py skill-candidate-run --force --limit 200
```

Optional mit Name:

```bash
python main.py skill-candidate-run --force --name "Text Normalisieren"
```

## Maintenance Manager Integration

`maintenance-run` führt die Pipeline als kontrollierten Wartungsschritt aus:

```bash
python main.py maintenance-run --force --limit 200
```

Der Maintenance Manager übernimmt dabei nur den Vorschlagsteil. Aktivierung bleibt ein separater User-Review-Schritt.

## Ergebnis

Neue Skill-Vorschläge landen unter:

```text
skill_proposals/
```

Ein Vorschlag enthält mindestens:

- `proposal.json`
- Skill-Metadaten
- Validierung
- Quelle `journal`

## Warum das wichtig ist

Damit Pandora nicht nur Tools erzeugt, sondern aus wiederkehrenden Abläufen echte Fähigkeiten ableiten kann. Das passt zum langfristigen Ziel: Wachstum wie ein lernendes System, aber mit klaren Sicherheitsgrenzen.
