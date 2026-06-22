# Obsidian Vault Integration

Pandora can use an Obsidian vault as an external Markdown knowledge source.

## Configuration

Set these values in `.env`:

```env
OBSIDIAN_VAULT_ENABLED=true
OBSIDIAN_VAULT_PATH=C:\Users\Thomas\Documents\Obsidian\MeinVault
OBSIDIAN_INBOX_DIR=Pandora_Inbox
OBSIDIAN_MODE=read_write_inbox_only
OBSIDIAN_CLOUD_ALLOWED=false
```

For Docker:

```env
OBSIDIAN_VAULT_PATH=/vault
```

and mount the vault:

```yaml
volumes:
  - "C:/Users/Thomas/Documents/Obsidian/MeinVault:/vault"
```

## Safety rules

Pandora may read Markdown files in the vault. Pandora may write only into the configured inbox directory.

Pandora must not:

- delete vault files
- move vault files
- overwrite existing vault files
- write outside `Pandora_Inbox`

## CLI

```bash
python main.py obsidian-status
python main.py obsidian-ensure-inbox
python main.py obsidian-index
python main.py obsidian-search "funktechnik"
python main.py obsidian-tags
python main.py obsidian-export --title "Neue Notiz" --content "Text" --category Knowledge --tag pandora
```

## API

```text
GET  /api/obsidian/status
POST /api/obsidian/reindex
GET  /api/obsidian/search?query=...
GET  /api/obsidian/tags
POST /api/obsidian/ensure-inbox
POST /api/obsidian/export
```

## Export metadata

Exports are written as Markdown with YAML frontmatter:

```yaml
---
title: Example
generated_by: pandora
generated_at: ...
review_status: pending
cloud_allowed: false
suggested_folder: Funktechnik/Messtechnik
tags:
  - pandora
---
```
