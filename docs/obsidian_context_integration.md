# MVP 23.5.4 – Obsidian Context Integration

Pandora kann Obsidian-Vault-Treffer jetzt als zusätzliche Knowledge-Quelle in den Chat-Kontext aufnehmen.

## Regeln

- `OBSIDIAN_VAULT_ENABLED=true` aktiviert die Quelle.
- `OBSIDIAN_VAULT_PATH` zeigt auf den Vault.
- `OBSIDIAN_CLOUD_ALLOWED=false` blockiert Obsidian-Kontext für Cloud-/Company-LLMs.
- Lokale LLMs dürfen Obsidian-Kontext verwenden, wenn der Vault aktiviert ist.
- Schreiben bleibt weiterhin nur in `Pandora_Inbox` erlaubt.

## CLI

```bash
python main.py obsidian-context-preview "Funktechnik" --provider-name local_fast
python main.py obsidian-context-preview "Funktechnik" --provider-name company_llm
```

## GUI

Die Seite `/obsidian-vault` enthält eine Context-Preview. Sie zeigt, ob Vault-Treffer in den Chat-Kontext gelangen würden oder wegen Policy blockiert werden.

## Chat-Ausführung

`knowledge_context` enthält zusätzlich:

- `blocked_obsidian_count`
- `obsidian.source_count`
- `obsidian.sources`
- Policy-Informationen
