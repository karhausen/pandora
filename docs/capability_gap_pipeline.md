# MVP 21.5 – Capability Gap Pipeline

The Capability Gap Pipeline turns repeated missing-capability signals into reviewable proposals.

It is intentionally observe-only:

- it reads the capability event log and task journal
- it clusters repeated missing abilities
- it writes proposal JSON files under `proposals/capability_gaps/`
- it does **not** generate code
- it does **not** install tools
- it does **not** activate skills
- it does **not** modify the core

## CLI

```bash
python main.py capability-gap-status
python main.py capability-gap-run --dry-run --force
python main.py capability-gap-run --force --limit 200
```

## Maintenance integration

`python main.py maintenance-run --force` now includes the capability gap pipeline before skill and tool improvement analysis.

## Purpose

This moves Pandora closer to the original growth vision: before creating new tools or skills, Pandora first identifies which abilities are repeatedly missing.
