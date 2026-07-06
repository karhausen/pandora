# Core Cleanup Note – MVP 30.4.3

Für diesen Cleanup wurde bewusst kein produktiver Core-Code gelöscht.

Entfernt wurden nur nicht benötigte Release-/Runtime-Artefakte:

- `.pytest_cache/`
- `core_before_cleanup/`

Grund: Der aktuelle Fokus ist Vault/LLM/Route-Registry-Stabilität. Ein aggressives Löschen alter Core-Module wäre jetzt riskant, weil API/CLI-Endpunkte teilweise noch historische Module importieren können.

Nächster Cleanup-Schritt erst nach stabilem Vault/LLM-Pfad:

1. Import-Graph erzeugen.
2. API-/CLI-Endpunkte gegenprüfen.
3. Legacy-Kandidaten markieren.
4. Erst danach verschieben oder löschen.
