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


## MVP 20.0.1 – Installation metadata normalization

Cloud-generated tools may define `TOOL_META` with design-oriented fields such as `tool_id` and may omit runtime fields such as `module`. Installation now normalizes metadata from three sources: proposal `spec`, proposal `design`, and generated module `TOOL_META`. The final runtime metadata always contains `id`, `module`, `function`, schemas, status, and security level before being written to the registry.


## MVP 20.2.2 – Output Contract Validation

Generated tools must honor their declared output schema. The deterministic generator and generated tests now recognize word-count-like designs even when the tool id is `word_counter` while the capability is `word_count`. For `output_schema={"count": "integer"}`, generated code must return `{"count": <int>}`.

Release cleanup also normalizes `config/tools/tool_registry.json` back to base tools so generated tools from local tests are not shipped in release ZIPs.
