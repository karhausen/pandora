# MVP 30.5 – Route Loop Stabilization

Dieser MVP erweitert MVP 30.4 um eine einfache Route-Schleife.

Der Router bleibt ein Dispatcher. Die fachliche Entscheidung trifft das LLM.

## Beispiel

User: `Welche Test-Prompts habe ich?`

1. LLM wählt `vault_search`.
2. Router ruft Vault/Knowledge-Kontext ab.
3. LLM sieht in der zweiten Planungsrunde, dass Kontext vorhanden ist.
4. LLM wählt `direct_answer`.
5. Antwort-LLM formuliert die Antwort mit Vault-Kontext.

## Begrenzung

Die Schleife ist auf maximal drei Runden begrenzt. Wiederholte identische Quellenrouten werden gestoppt.
