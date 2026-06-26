# MVP 25.1.3 – Frontmatter Governance & Validation

Pandora now validates Obsidian YAML frontmatter with PyYAML instead of a hand-written parser.

## Goals

- Never crash indexing/import candidate generation because one Markdown note has invalid frontmatter.
- Surface invalid YAML as governance warnings.
- Provide CLI/API validation entry points.

## CLI

```bash
python main.py obsidian-validate
```

## API

```text
GET /api/obsidian/frontmatter/validate
```

## Behavior

Invalid frontmatter is returned as metadata with `_frontmatter_valid: false` and a readable error message. The file can still be indexed, but tags/policy fields from invalid YAML are not trusted.

Example valid frontmatter:

```yaml
---
tags:
  - docs
  - notiz
  - pandora
cloud_allowed: false
company_allowed: true
---
```
