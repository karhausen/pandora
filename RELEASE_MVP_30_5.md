# MVP 30.5 – Route Loop Stabilization

Ziel: Der LLM-geführte Router bleibt Dispatcher, kann aber mehrere kontrollierte Runden fahren.

## Ablauf

1. Prompt Builder zeigt dem LLM verfügbare Routen.
2. LLM fordert eine Route an, z. B. `vault_search`.
3. Python validiert und führt die Route aus.
4. Das Ergebnis wird dem Route Planner in der nächsten Runde als bereits ausgeführte Route angezeigt.
5. Wenn genug Kontext vorhanden ist, wählt das LLM `direct_answer`.
6. Erst dann erzeugt das Antwort-LLM die User-Antwort mit gesammeltem Kontext.

## Aktiv

- `direct_answer`
- `vault_search`
- `memory_search`
- `clarify_user`

## Weiterhin deaktiviert

- Tools
- Skills
- Capability Gap
- Tool Factory
- Evolution

## Sicherheitsregel

Der Router entscheidet weiterhin nicht anhand der User-Anfrage. Er dispatcht ausschließlich die vom LLM angeforderte Route.
