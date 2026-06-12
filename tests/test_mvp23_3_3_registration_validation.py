import subprocess
import sys

from core.registration_validator import RegistrationValidator


def test_registration_validator_cli_handlers_are_complete():
    report = RegistrationValidator().validate_cli()
    assert report["ok"] is True
    assert report["missing_handlers"] == []
    assert report["command_count"] > 0


def test_registration_validator_full_report_is_ok_enough():
    report = RegistrationValidator().validate()
    assert report["checks"]["cli"]["ok"] is True
    assert report["checks"]["api"]["ok"] is True
    assert report["error_count"] == 0


def test_registration_validate_cli_command_runs():
    result = subprocess.run([sys.executable, "main.py", "registration-validate"], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "registration_validation_report" in result.stdout
    assert "missing_handlers" in result.stdout


def test_registration_validation_api_route_exists():
    from core.api import app
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/system/registration-validation" in paths
    assert "/api/system/registration-validation/cli" in paths
