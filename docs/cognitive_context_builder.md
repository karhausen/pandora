# MVP 25.1 – Cognitive Context Builder

Pandora now builds policy-safe context before calling an LLM.

## Policy

- local: local LLMs may use local-only user knowledge and Obsidian context.
- company: company LLMs may use company-allowed context only.
- cloud: public cloud LLMs may use cloud-allowed context only.

Obsidian defaults are private:

```env
OBSIDIAN_COMPANY_ALLOWED=false
OBSIDIAN_CLOUD_ALLOWED=false
```

Per Obsidian note frontmatter can define:

```yaml
company_allowed: true
cloud_allowed: false
```

## Commands

```bash
python main.py cognitive-context-status
python main.py cognitive-context-preview "Was sind die Topics in meinem Vault?"
```

If the active route is company_llm and Obsidian is not company-allowed, Pandora returns a clear policy explanation instead of letting the LLM claim it has no Vault access.
