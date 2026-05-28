from pathlib import Path
from core.documentation_generator import DocumentationGenerator
from core.governance import Governance
from core.changelog_manager import ChangelogManager


def test_docs_generate():
    result = DocumentationGenerator().generate()
    assert result["generated"] is True
    assert Path("docs/architecture.md").exists()
    assert Path("README.md").exists()


def test_governance_check():
    result = Governance().check()
    assert "ok" in result
    assert "issues" in result


def test_changelog():
    content = ChangelogManager().read()
    assert "Changelog" in content
