# MVP 28.8 – Improvement Prioritization

Pandora bewertet erkannte Patterns als Verbesserungskandidaten. Diese Stufe erzeugt keine Proposals und aktiviert keine Änderungen.

## CLI

```bash
python main.py improvement-priority-status
python main.py improvement-priority-candidates --limit 100
python main.py improvement-priority-prioritize --limit 100 --save
python main.py improvement-priority-queue
python main.py improvement-priority-history
python main.py improvement-priority-weights
```

## Prinzip

Observation sammelt Fakten, Pattern Recognition erkennt Muster, Improvement Prioritization bewertet Nutzen, Risiko, Aufwand, Confidence, Häufigkeit, Dringlichkeit und Benutzerwert. Erst ein späterer Schritt darf daraus reviewbare Proposals erzeugen.
