# MVP 22.8 – Knowledge Search & Context Injection

Pandora kann die `user_knowledge/`-Ablage durchsuchen und daraus einen begrenzten, policy-sicheren Kontext für Chat-Antworten bauen.

## Regeln

- `public/` darf lokal und in Cloud-Kontext verwendet werden.
- `restricted_cloud_allowed/` darf nach Policy-Prüfung in Cloud-Kontext.
- `private_local_only/` wird nur bei lokalen Zielen injiziert.
- Bei Cloud-/Company-Routen wird `private_local_only/` gezählt, aber nicht in den Prompt übernommen.

## API

- `GET /api/gui/knowledge/search`
- `GET /api/gui/knowledge/context-preview`
- `GET /api/gui/knowledge/context-injection-preview`

## Chat-Integration

`ChatService` ruft `KnowledgeContextService.build_for_chat()` auf. Der Service löst die aktive Chat-Route auf und entscheidet daraus, ob der Zielkontext lokal oder cloud/company ist.

Der Ausführungsbericht enthält:

```json
{
  "knowledge_context": {
    "source_count": 1,
    "sources": [],
    "target": "local",
    "cloud_context": false,
    "blocked_local_only_count": 0
  }
}
```

Wichtig: Inhalte werden begrenzt und mit Quellenkopf injiziert. Das ist kein Vektorindex, sondern eine einfache, nachvollziehbare MVP-Suche.
