from __future__ import annotations

import importlib
import json
import shutil
from datetime import datetime, UTC
from pathlib import Path

from .config import GENERATED_TOOLS_DIR, TOOL_ACTIVATION_LOG
from .models import SecurityLevel, ToolActivationResult, ToolMeta, ToolStatus
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
            raw_meta = getattr(module, "TOOL_META", None)
            meta = self._normalize_tool_meta(raw_meta, proposal=proposal, tool_id=tool_id)
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


    def _normalize_tool_meta(self, raw_meta, proposal: dict, tool_id: str) -> ToolMeta:
        """Build a valid ToolMeta from generated code plus proposal metadata.

        Cloud-generated tools sometimes use design-style keys such as
        ``tool_id`` and omit the runtime-only ``module`` field. Installation is
        the boundary where Pandora knows the final module path, so normalize the
        metadata here instead of rejecting an otherwise valid proposal.
        """
        spec = dict(proposal.get("spec") or {})
        design = dict(proposal.get("design") or {})
        raw = dict(raw_meta or {}) if isinstance(raw_meta, dict) else {}

        merged = {**spec, **design, **raw}
        normalized_id = merged.get("id") or merged.get("tool_id") or spec.get("id") or tool_id
        aliases = set(merged.get("aliases") or [])
        capability = proposal.get("capability") or spec.get("capability") or design.get("capability")
        if capability and capability != normalized_id:
            aliases.add(capability)
        if tool_id and tool_id != normalized_id:
            aliases.add(tool_id)

        normalized = {
            "id": normalized_id,
            "name": merged.get("name") or spec.get("name") or tool_id.replace("_", " ").title(),
            "description": merged.get("description") or spec.get("description") or f"Generated tool: {tool_id}",
            "version": merged.get("version") or "0.1.0",
            "input_schema": merged.get("input_schema") or spec.get("input_schema") or {},
            "output_schema": merged.get("output_schema") or spec.get("output_schema") or {},
            "security_level": merged.get("security_level") or spec.get("security_level") or SecurityLevel.SAFE.value,
            "status": merged.get("status") or ToolStatus.ACTIVE.value,
            "module": f"generated_tools.{tool_id}",
            "function": merged.get("function") or "run",
            "aliases": sorted(aliases),
            "installed_from": proposal.get("id"),
        }
        return ToolMeta.model_validate(normalized)

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
        if tool_id in {"word_count", "word_count_tool", "word_counter"}:
            return {"text": "eins zwei drei"}
        if tool_id == "timestamp":
            return {}
        return {"text": "hello"}
