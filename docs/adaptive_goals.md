# MVP 29.2 – Adaptive Goals

## Ziel

Adaptive Goals fuehrt langfristige, kontrollierte Ziele fuer Pandora ein. Diese Ziele steuern die Evolution nicht automatisch, sondern dienen als transparente Entscheidungsgrundlage fuer kuenftige Proposals.

## Prinzipien

- Ziele duerfen keine Core-Aenderungen ausfuehren.
- Ziele duerfen keine Tools aktivieren.
- Repriorisierung ist Metadatenpflege, keine Umsetzung.
- Jede konkrete Verbesserung laeuft weiterhin ueber EvolutionProposal, Review, Tests und Benutzerfreigabe.

## CLI

```powershell
python main.py goals status
python main.py goals list
python main.py goals show goal_evolution_quality
python main.py goals history
python main.py goals evaluate
python main.py goals reprioritize
python main.py goals reprioritize --write
```

## API

```text
GET  /api/goals/status
GET  /api/goals/list
GET  /api/goals/show/{goal_id}
GET  /api/goals/history
GET  /api/goals/evaluate
POST /api/goals/reprioritize
```

## GUI

```text
/adaptive-goals
```

## Ergebnis

Pandora kann langfristige Ziele anzeigen, bewerten, historisieren und priorisieren. Die Aktivierung konkreter Verbesserungen bleibt strikt reviewpflichtig.
