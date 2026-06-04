# Controlled Tool Factory

MVP 20.0 closes the controlled tool-development loop.

```text
Capability Gap
↓
Tool Design
↓
Cloud Code Generation
↓
Policy Review
↓
Offline Tests
↓
Proposal VALIDATED
↓
Manual APPROVE
↓
INSTALL
↓
Tool Registry
↓
Tool usable by Planner/Worker
```

## Lifecycle

- `VALIDATED`: code and tests passed, but tool is not active.
- `APPROVED`: user approved installation.
- `REJECTED`: user rejected the proposal.
- `INSTALLED`: code copied into `generated_tools/` and registry updated.

Installation is refused unless the proposal is `APPROVED`.

## CLI

```bash
python3 main.py proposal-list
python3 main.py proposal-show <PROPOSAL_ID>
python3 main.py proposal-approve <PROPOSAL_ID>
python3 main.py proposal-reject <PROPOSAL_ID>
python3 main.py proposal-install <PROPOSAL_ID> --test-json '{"text":"eins zwei drei"}'
python3 main.py tool-list
```

## Safety

Cloud-generated code is never activated directly. It must pass local review and tests, then be approved and installed explicitly.
