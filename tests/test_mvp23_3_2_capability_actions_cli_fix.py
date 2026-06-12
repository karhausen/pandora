import subprocess
import sys


def test_main_parser_builds_with_capability_action_commands():
    result = subprocess.run([sys.executable, "main.py", "--help"], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0
    assert "capability-actions-status" in result.stdout
    assert "capability-actions-rebuild" in result.stdout
    assert "capability-action-show" in result.stdout


def test_capability_actions_status_cli_runs():
    result = subprocess.run([sys.executable, "main.py", "capability-actions-status"], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "capability_action_status" in result.stdout
    assert "requires_user_approval" in result.stdout
