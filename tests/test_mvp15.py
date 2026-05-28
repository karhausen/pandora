from core.execution_policy import ExecutionPolicyManager
from core.permission_manager import PermissionManager
from core.sandbox import Sandbox
from core.tool_registry import ToolRegistry


def test_execution_policy_for_calculator():
    policy = ExecutionPolicyManager().get_for_tool("calculator")
    assert policy.name.value == "isolated"
    assert policy.timeout > 0


def test_permission_manager_blocks_shell():
    policy = ExecutionPolicyManager().get_for_tool("calculator")
    review = PermissionManager().review_code("import subprocess\nsubprocess.run(['x'])", policy)
    assert review["ok"] is False


def test_sandbox_run_calculator():
    registry = ToolRegistry()
    registry.discover()
    result = Sandbox().run_tool("calculator", {"expression": "2+3*4"})
    assert result["success"] is True
    assert result["output"]["result"] == 14
    assert result["isolated"] is True


def test_sandbox_logs():
    logs = Sandbox().logs()
    assert isinstance(logs, list)
