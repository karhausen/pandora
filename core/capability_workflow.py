from __future__ import annotations

import uuid
from datetime import datetime, UTC

from .agent_loop import AgentLoop
from .capability_expansion_manager import CapabilityExpansionManager
from .capability_workflow_log import CapabilityWorkflowLog
from .models import CapabilityWorkflowResult
from .tool_activation_manager import ToolActivationManager


class CapabilityWorkflow:
    def __init__(self):
        self.expansion = CapabilityExpansionManager()
        self.activation = ToolActivationManager()
        self.log = CapabilityWorkflowLog()

    async def propose_only(self, task: str) -> CapabilityWorkflowResult:
        return await self.run(task, activate=False, retry=False, mode="propose_only")

    async def propose_activate(self, task: str, retry: bool = False) -> CapabilityWorkflowResult:
        return await self.run(task, activate=True, retry=retry, mode="propose_activate_retry" if retry else "propose_activate")

    async def run(self, task: str, activate: bool = False, retry: bool = False, mode: str = "manual") -> CapabilityWorkflowResult:
        workflow_id = f"cw_{uuid.uuid4().hex[:12]}"
        proposal_id = None
        activation_result = None
        retry_result = None
        error = None
        success = False

        try:
            expansion = self.expansion.evaluate_task(task, auto_propose=True)
            proposal = expansion.get("proposal")
            if proposal:
                proposal_id = proposal["id"]

            if not proposal_id:
                result = CapabilityWorkflowResult(
                    workflow_id=workflow_id,
                    task=task,
                    success=False,
                    mode=mode,
                    proposal_created=False,
                    error="No capability gap detected or no proposal created.",
                    created_at=datetime.now(UTC).isoformat(),
                )
                self.log.append(result.model_dump(mode="json"))
                return result

            activated = False
            if activate:
                activation = await self.activation.activate(proposal_id)
                activation_result = activation.model_dump(mode="json")
                activated = bool(activation.activated)

                if retry and activated:
                    retry_run = await AgentLoop().run(task, provider_name="mock")
                    retry_result = retry_run.model_dump(mode="json")
                    success = bool(retry_run.success)
                else:
                    success = activated
            else:
                activated = False
                success = True

            result = CapabilityWorkflowResult(
                workflow_id=workflow_id,
                task=task,
                success=success,
                mode=mode,
                proposal_created=True,
                proposal_id=proposal_id,
                activated=activated,
                activation=activation_result,
                retry_result=retry_result,
                error=error,
                created_at=datetime.now(UTC).isoformat(),
            )
            self.log.append(result.model_dump(mode="json"))
            return result

        except Exception as exc:
            result = CapabilityWorkflowResult(
                workflow_id=workflow_id,
                task=task,
                success=False,
                mode=mode,
                error=f"{type(exc).__name__}: {exc}",
                created_at=datetime.now(UTC).isoformat(),
            )
            self.log.append(result.model_dump(mode="json"))
            return result

    def list(self, limit: int = 20) -> list[dict]:
        return self.log.list(limit)

    def last(self) -> dict | None:
        return self.log.last()
