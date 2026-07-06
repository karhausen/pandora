# MVP 30.4.1 – Route Planner Provider Fix

Goal: keep MVP 30.4 focused on the LLM-led route registry and remove the unsafe mock fallback from live route selection.

## Fixed

- The route planner now uses the active profile `cloud_expert` when no explicit provider is selected.
- Route selection disables provider fallback so the mock LLM cannot silently choose `clarify_user` for real chat requests.
- The router remains a dispatcher only. It still does not inspect user text or decide whether Vault, Memory, or direct answer is needed.

## Scope not changed

- No tools.
- No skills.
- No capability gap.
- No tool factory.
