from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.release_audit import audit
from scripts.export_release import copy_source, sanitize_tree, make_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_release_audit_detects_runtime_and_secret_files(tmp_path: Path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "x.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-testsecret123456789012345\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"cache")

    result = audit(tmp_path)

    assert result["ok"] is False
    paths = {issue["path"] for issue in result["issues"]}
    assert ".env" in paths
    assert "__pycache__" in paths or "__pycache__/x.pyc" in paths


def test_release_audit_accepts_example_secret_files(tmp_path: Path):
    (tmp_path / "config" / "llm").mkdir(parents=True)
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=your_key_here\n", encoding="utf-8")
    (tmp_path / "config" / "llm" / "llm_config.local.example.json").write_text('{"api_key": "env:OPENAI_API_KEY"}\n', encoding="utf-8")

    result = audit(tmp_path)

    assert result["ok"] is True


def test_sanitize_tree_resets_runtime_content(tmp_path: Path):
    package = tmp_path / "package"
    copy_source(ROOT, package)
    (package / "logs" / "run.log").parent.mkdir(parents=True, exist_ok=True)
    (package / "logs" / "run.log").write_text("runtime", encoding="utf-8")
    (package / "config" / "llm" / "llm_config.local.json").parent.mkdir(parents=True, exist_ok=True)
    (package / "config" / "llm" / "llm_config.local.json").write_text("{}", encoding="utf-8")

    sanitize_tree(package)
    result = audit(package)

    assert not (package / "logs" / "run.log").exists()
    assert (package / "logs" / ".gitkeep").exists()
    assert result["ok"] is True


def test_manifest_contains_hashes(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "README.md").write_text("Pandora\n", encoding="utf-8")
    audit_result = {"ok": True, "issues": []}

    manifest = make_manifest(package, "test-version", audit_result)
    data = json.loads((package / "release_manifest.json").read_text(encoding="utf-8"))

    assert manifest["sanitized"] is True
    assert data["version"] == "test-version"
    assert data["files"][0]["path"] == "README.md"
    assert len(data["files"][0]["sha256"]) == 64
