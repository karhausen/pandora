from __future__ import annotations

import importlib
import json
import shutil
from datetime import datetime, UTC
from pathlib import Path

from .config import GENERATED_TOOLS_DIR, TOOL_ACTIVATION_LOG
from .models import ToolActivationResult, ToolMeta
from .tool_executor import ToolExecutor
from .tool_proposal_manager import ToolProposalManager
from .tool_registry import ToolRegistry


class ToolActivationManager:
    def __init__(self):
        self.proposals = ToolProposalManager()
        self.registry = ToolRegistry()
        TOOL_ACTIVATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    async def activate(self, proposal_id: str, test_payload: dict | None = None) -> ToolActivationResult:
        try:
            shown = self.proposals.show(proposal_id)
            proposal = shown["proposal"]
            if proposal["status"] != "APPROVED":
                return self._record(ToolActivationResult(
                    activated=False,
                    proposal_id=proposal_id,
                    error="Only APPROVED proposals can be installed.",
                ))

            spec = proposal["spec"]
            tool_id = spec["id"]
            src = Path(proposal["code_file"])
            GENERATED_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
            init = GENERATED_TOOLS_DIR / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")

            dst = GENERATED_TOOLS_DIR / f"{tool_id}.py"
            shutil.copy2(src, dst)

            importlib.invalidate_caches()
            module = importlib.import_module(f"generated_tools.{tool_id}")
            meta = ToolMeta.model_validate(getattr(module, "TOOL_META"))
            self.registry.register(meta)

            tested = False
            test_payload = test_payload if test_payload is not None else self._default_payload(tool_id)
            result = await ToolExecutor(self.registry).run_tool(tool_id, test_payload)
            tested = bool(result.success)
            if not tested:
                return self._record(ToolActivationResult(
                    activated=False,
                    proposal_id=proposal_id,
                    tool_id=tool_id,
                    copied_to=str(dst),
                    registered=True,
                    tested=False,
                    error=result.error or "Activation test failed.",
                ))

            result = self._record(ToolActivationResult(
                activated=True,
                proposal_id=proposal_id,
                tool_id=tool_id,
                copied_to=str(dst),
                registered=True,
                tested=True,
            ))
            self.proposals.mark_installed(proposal_id, activation=result.model_dump(mode="json"))
            return result

        except Exception as exc:
            return self._record(ToolActivationResult(
                activated=False,
                proposal_id=proposal_id,
                error=f"{type(exc).__name__}: {exc}",
            ))

    def list_log(self, limit: int = 20) -> list[dict]:
        if not TOOL_ACTIVATION_LOG.exists():
            return []
        lines = TOOL_ACTIVATION_LOG.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def _record(self, result: ToolActivationResult) -> ToolActivationResult:
        entry = result.model_dump(mode="json")
        entry["created_at"] = datetime.now(UTC).isoformat()
        with TOOL_ACTIVATION_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result

    def _default_payload(self, tool_id: str) -> dict:
        if tool_id == "json_pretty":
            return {"text": "{\"b\":2,\"a\":1}"}
        if tool_id == "text_reverse":
            return {"text": "abc"}
        if tool_id in {"word_count", "word_count_tool"}:
            return {"text": "eins zwei drei"}
        if tool_id == "timestamp":
            return {}
        return {"text": "hello"}
