# MVP 23.5.3 – Obsidian Vault GUI

Die Obsidian-Integration ist jetzt im Knowledge-Bereich über `/obsidian-vault` bedienbar.

## Funktionen

- Status des konfigurierten Vaults anzeigen
- `Pandora_Inbox` sicher anlegen
- Vault indexieren
- Markdown-Dateien durchsuchen
- Tags und Wikilinks in Treffern anzeigen
- Markdown-Entwürfe nach `Pandora_Inbox` exportieren
- Inbox-Einträge anzeigen und Review-Status setzen

## Sicherheitsregeln

Pandora schreibt weiterhin nur in das konfigurierte Inbox-Verzeichnis. Löschen, Verschieben und Überschreiben im Vault bleiben verboten.

Konfiguration erfolgt lokal über `.env`:

```env
OBSIDIAN_VAULT_ENABLED=true
OBSIDIAN_VAULT_PATH=/path/to/vault
OBSIDIAN_INBOX_DIR=Pandora_Inbox
OBSIDIAN_MODE=read_write_inbox_only
OBSIDIAN_CLOUD_ALLOWED=false
```
