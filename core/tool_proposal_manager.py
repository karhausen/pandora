from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, UTC
from pathlib import Path

from .capability_detector import CapabilityDetector
from .config import ROOT_DIR, TOOL_PROPOSALS_DIR
from .llm_tool_generator import LLMToolGenerator
from .models import (
    ToolGenerationAttempt,
    ToolGenerationResult,
    ToolProposal,
    ToolProposalStatus,
)
from .tool_generation_log import ToolGenerationLog
from .tool_generation_runner import ToolGenerationRunner
from .tool_generator import ToolGenerator
from .tool_repair_manager import ToolRepairManager
from .tool_test_generator import ToolTestGenerator
from .tool_validator import ToolValidator


class ToolProposalManager:
    def __init__(self):
        self.root = TOOL_PROPOSALS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.generator = ToolGenerator()
        self.test_generator = ToolTestGenerator()
        self.validator = ToolValidator()
        self.llm_generator = LLMToolGenerator()
        self.repair_manager = ToolRepairManager()
        self.generation_runner = ToolGenerationRunner()
        self.generation_log = ToolGenerationLog()

    def detect_gap(self, task: str, analysis: dict | None = None) -> dict:
        return CapabilityDetector().detect(task, analysis=analysis)

    def propose_for_capability(self, capability: str) -> dict:
        spec = self.generator.build_spec(capability)
        proposal_id = self._new_id()
        proposal_dir, tool_dir, test_dir = self._create_proposal_dirs(proposal_id)

        code = self.generator.generate_code(spec)
        test_code = self.test_generator.generate_test(spec)

        code_file = tool_dir / f"{spec.id}.py"
        test_file = test_dir / f"test_{spec.id}.py"
        code_file.write_text(code, encoding="utf-8")
        test_file.write_text(test_code, encoding="utf-8")

        static = self.validator.static_review(code)
        test_result = self.validator.run_tests(proposal_dir) if static["ok"] else {"success": False, "skipped": True}
        validation = {"static": static, "tests": test_result}
        status = ToolProposalStatus.VALIDATED if static["ok"] and test_result.get("success") else ToolProposalStatus.FAILED

        proposal = self._write_proposal(
            proposal_id=proposal_id,
            capability=capability,
            spec=spec,
            proposal_dir=proposal_dir,
            code_file=code_file,
            test_file=test_file,
            validation=validation,
            status=status,
            risk=static["risk"],
        )
        return proposal.model_dump(mode="json")

    def generate_with_llm(
        self,
        capability: str,
        provider_name: str | None = "mock",
        model: str | None = None,
        max_attempts: int = 2,
        run_tests: bool = True,
    ) -> dict:
        spec = self.generator.build_spec(capability)
        proposal_id = self._new_id()
        proposal_dir, tool_dir, test_dir = self._create_proposal_dirs(proposal_id)

        code_file = tool_dir / f"{spec.id}.py"
        test_file = test_dir / f"test_{spec.id}.py"
        test_file.write_text(self.test_generator.generate_test(spec), encoding="utf-8")

        attempts: list[ToolGenerationAttempt] = []
        previous_error = None
        best_validation = {}
        status = ToolProposalStatus.FAILED

        for attempt_no in range(1, max_attempts + 1):
            generated = (
                self.llm_generator.generate_code(spec, provider_name=provider_name, model=model)
                if attempt_no == 1
                else self.repair_manager.repair(spec, previous_error or "Unknown error", provider_name=provider_name, model=model)
            )
            code = generated["code"]
            code_file.write_text(code, encoding="utf-8")

            static = self.validator.static_review(code)
            if static["ok"] and run_tests:
                test_result = self.generation_runner.run_pytest(proposal_dir)
            elif static["ok"] and not run_tests:
                test_result = {"success": True, "skipped": True, "reason": "Tests skipped by caller."}
            else:
                test_result = {"success": False, "skipped": True, "stderr": "\n".join(static["issues"])}

            success = bool(static["ok"] and test_result.get("success"))
            previous_error = None if success else (test_result.get("stderr") or test_result.get("stdout") or "Validation failed")
            best_validation = {
                "static": static,
                "tests": test_result,
                "source": generated.get("source"),
                "llm_used": generated.get("llm_used"),
            }
            attempts.append(ToolGenerationAttempt(
                attempt=attempt_no,
                success=success,
                code_review=static,
                test_result=test_result,
                error=previous_error,
            ))

            if success:
                status = ToolProposalStatus.VALIDATED
                break

        proposal = self._write_proposal(
            proposal_id=proposal_id,
            capability=capability,
            spec=spec,
            proposal_dir=proposal_dir,
            code_file=code_file,
            test_file=test_file,
            validation={
                "attempts": [a.model_dump(mode="json") for a in attempts],
                "latest": best_validation,
            },
            status=status,
            risk=best_validation.get("static", {}).get("risk", "HIGH"),
        )

        generation = ToolGenerationResult(
            success=status == ToolProposalStatus.VALIDATED,
            capability=capability,
            proposal_created=True,
            proposal_id=proposal_id,
            attempts=attempts,
            error=None if status == ToolProposalStatus.VALIDATED else previous_error,
        )
        self.generation_log.append(generation.model_dump(mode="json"))

        return {
            "generation": generation.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
        }

    def propose_from_task(self, task: str, analysis: dict | None = None) -> dict:
        gap = self.detect_gap(task, analysis=analysis)
        if not gap.get("gap_detected"):
            return {"created": False, "gap": gap, "proposal": None}
        proposal = self.propose_for_capability(gap["capability"])
        return {"created": True, "gap": gap, "proposal": proposal}

    def list(self) -> list[dict]:
        proposals = []
        for path in sorted(self.root.glob("tool_*"), reverse=True):
            p = path / "proposal.json"
            if p.exists():
                proposals.append(json.loads(p.read_text(encoding="utf-8")))
        return proposals

    def show(self, proposal_id: str) -> dict:
        path = self.root / proposal_id
        if not path.exists():
            raise FileNotFoundError(proposal_id)
        result = {"path": str(path)}
        for name in ["proposal.json", "validation.json"]:
            p = path / name
            if p.exists():
                result[name.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))
        return result

    def prepare_activation_copy(self, proposal_id: str) -> dict:
        data = self.show(proposal_id)
        proposal = data["proposal"]
        if proposal["status"] != "VALIDATED":
            return {"prepared": False, "error": "Only validated tool proposals can be prepared."}

        src = Path(proposal["code_file"])
        dst = ROOT_DIR / "generated_tools" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        init = dst.parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
        return {"prepared": True, "copied_to": str(dst), "note": "Copied only. Not registered automatically."}

    def _new_id(self) -> str:
        return f"tool_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _create_proposal_dirs(self, proposal_id: str):
        proposal_dir = self.root / proposal_id
        tool_dir = proposal_dir / "generated_tools"
        test_dir = proposal_dir / "tests"
        tool_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        (tool_dir / "__init__.py").write_text("", encoding="utf-8")
        (proposal_dir / "pytest.ini").write_text("[pytest]\npythonpath = .\n", encoding="utf-8")
        return proposal_dir, tool_dir, test_dir

    def _write_proposal(
        self,
        proposal_id,
        capability,
        spec,
        proposal_dir,
        code_file,
        test_file,
        validation,
        status,
        risk,
    ):
        proposal = ToolProposal(
            id=proposal_id,
            status=status,
            capability=capability,
            spec=spec,
            created_at=datetime.now(UTC).isoformat(),
            proposal_dir=str(proposal_dir),
            code_file=str(code_file),
            test_file=str(test_file),
            validation=validation,
            risk=risk,
        )
        (proposal_dir / "proposal.json").write_text(json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        (proposal_dir / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
        return proposal
