#!/usr/bin/env python3
"""MVP 30.11 Legacy Quarantine Dry Run.
Reads reports/core_triage_report_mvp30_10.json and reports/core_runtime_analysis_mvp30_9.json if present.
Does not move/delete files.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
triage_path = ROOT / "reports" / "core_triage_report_mvp30_10.json"
analysis_path = ROOT / "reports" / "core_runtime_analysis_mvp30_9.json"
out_path = ROOT / "reports" / "legacy_quarantine_dry_run_mvp30_11.json"

if not triage_path.exists():
    raise SystemExit(f"Missing {triage_path}")
if not analysis_path.exists():
    raise SystemExit(f"Missing {analysis_path}")

triage = json.loads(triage_path.read_text(encoding="utf-8"))
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
modules = {m["module"]: m for m in analysis.get("modules", [])}
checks = []
for item in triage.get("triage", []):
    if item.get("category") != "D":
        continue
    mod = item["module"]
    m = modules.get(mod, {})
    imported_by = m.get("imported_by", [])
    importing_reachable = [ib for ib in imported_by if modules.get(ib, {}).get("reachable_from_entrypoints")]
    risk = "low" if not imported_by else ("medium" if not importing_reachable else "high")
    checks.append({
        "path": item["path"],
        "module": mod,
        "target_path": "legacy/" + item["path"].replace("\\", "/"),
        "imported_by": imported_by,
        "reachable_from_entrypoints": m.get("reachable_from_entrypoints", False),
        "static_import_break_risk": risk,
        "would_move": True,
        "notes": item.get("reason", ""),
    })

payload = {
    "kind": "mvp30_11_legacy_quarantine_dry_run",
    "rules": [
        "DRY RUN only: no runtime file moved or deleted.",
        "Only category D from MVP 30.10 is simulated.",
        "Static import graph only: dynamic imports are not proven safe.",
    ],
    "candidate_count": len(checks),
    "risk_counts": {r: sum(1 for c in checks if c["static_import_break_risk"] == r) for r in ["low", "medium", "high"]},
    "candidates": checks,
}
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"written": str(out_path), "candidate_count": len(checks), "risk_counts": payload["risk_counts"]}, indent=2))
