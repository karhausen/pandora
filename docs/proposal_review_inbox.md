# MVP 21.6 – Proposal Review Inbox

The Proposal Review Inbox is the central morning inbox for Pandora's controlled growth.

It scans reviewable JSON output from:

- nightly governance reviews
- maintenance reports
- capability gap proposals
- skill candidate proposals
- tool improvement proposals
- tool proposals
- core improvement proposals

## Principle

The inbox is **observe-first**.

It may:

- scan proposal/report files
- summarize pending review items
- show one item in detail
- write a small `review_state.json` when the user marks an item reviewed

It must not:

- activate a tool
- activate a skill
- change core source files
- run generated code
- call the network
- modify credentials or profiles

## CLI

```bash
python main.py review-inbox-status
python main.py review-inbox-list
python main.py review-inbox-show <item_id>
python main.py review-inbox-mark <item_id> --decision reviewed --note "checked"
```

Supported decisions:

- `reviewed`
- `accepted_for_next_step`
- `rejected`
- `needs_work`

This keeps the long-term Variante C vision practical: Pandora can reorganize and prepare review packages at night, but the morning decision stays visible and controlled.
