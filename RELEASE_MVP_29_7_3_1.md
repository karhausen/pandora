# MVP 29.7.3.1 – Known Capability Preflight Fix

## Ziel

Bekannte Pandora-Systemfähigkeiten wie Obsidian, Vault, Knowledge und Memory dürfen nicht von der Semantic Capability Decision Engine blockiert oder fälschlich in Tool-Development umgeleitet werden.

## Änderungen

- Neue Preflight-Prüfung für bekannte Systemfähigkeiten in `ChatResponseRouter`.
- `CoordinatorAgent` routet Obsidian-/Vault-Anfragen wieder in den Chat-/Knowledge-Pfad statt in `tool_development`.
- `ChatService` überspringt die Semantic Capability Decision Engine bei bekannten Systemfähigkeiten.
- Obsidian-Übersichtsfragen wie `Was steht in meinem Obsidian-Vault?` werden wieder als Vault-/Knowledge-Anfragen behandelt.
- Obsidian-Konfigurationsprobleme werden direkt und verständlich gemeldet, statt als Capability-Gap-Fehler zu erscheinen.
- Integration-Selftest prüft jetzt, dass Obsidian-Vault-Anfragen nicht in `tool_development` landen.

## Architekturregel

Die Semantic Capability Decision Engine ist für unklare oder fehlende Fähigkeiten zuständig. Sie darf bekannte, deterministische Pandora-Systemfähigkeiten nicht blockieren.

## Geprüft

- `python main.py selftest integration`: OK
- Syntaxprüfung der geänderten Module: OK
