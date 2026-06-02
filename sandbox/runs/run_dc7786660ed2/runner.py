from __future__ import annotations
import importlib
import json
import sys
import traceback

sys.path.insert(0, r"/mnt/data/pandora_19_2_work/pandora_agent_mvp_19_0_2")
payload_file = r"/mnt/data/pandora_19_2_work/pandora_agent_mvp_19_0_2/sandbox/runs/run_dc7786660ed2/payload.json"
result_file = r"/mnt/data/pandora_19_2_work/pandora_agent_mvp_19_0_2/sandbox/runs/run_dc7786660ed2/result.json"

try:
    payload = json.loads(open(payload_file, "r", encoding="utf-8").read())
    module = importlib.import_module("tools.calculator")
    fn = getattr(module, "run")
    output = fn(payload)
    result = {"success": True, "output": output, "error": None}
except Exception as exc:
    result = {"success": False, "output": None, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}

open(result_file, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False))
