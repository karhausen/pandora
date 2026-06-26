---
cloud_allowed: false
company_allowed: true

tags:
  #docs #commands
---

# Wichtige Befehle

## Start

```bash
python main.py status
python main.py api --host 127.0.0.1 --port 8000
```

## [[Core]]

```bash
python main.py control-status
python main.py heartbeat
```

## [[Maintenance]] und Night Mode

```bash
python main.py maintenance-status
python main.py maintenance-run --dry-run --force
python main.py operations-preview
python main.py operations-run --force
```

## Review und Approval

```bash
python main.py review-inbox-list
python main.py review-inbox-show <item_id>
python main.py review-inbox-mark <item_id> --decision reviewed
python main.py approval-pending
python main.py approval-decide <item_id> --decision approve_next_step --note "geprüft"
python main.py approval-audit
```

## Tools

```bash
python main.py tool-list
python main.py tool-info <tool_id>
python main.py tool-enable <tool_id>
python main.py tool-disable <tool_id>
python main.py tool-improvement-status
python main.py tool-improvement-run --dry-run --force
```

## Skills

```bash
python main.py skill-candidate-status
python main.py skill-candidate-run --dry-run --force
```

## Knowledge

```bash
python main.py knowledge-governance-status
python main.py knowledge-governance-run
python main.py knowledge-metadata-audit
```

## Tests und Release

```bash
python -m pytest -q
python -m compileall -q .
python scripts/release_audit.py .
python scripts/export_release.py --skip-tests
```
