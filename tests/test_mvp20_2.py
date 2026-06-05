from __future__ import annotations

from fastapi.testclient import TestClient

from core.api import app


def test_user_gui_contains_tool_factory_workflow_controls():
    html = (PathLike := __import__('pathlib').Path)('web/index.html').read_text(encoding='utf-8')
    js = PathLike('web/user.js').read_text(encoding='utf-8')

    assert 'id="toolWorkflow"' in html
    assert 'Approve' in html
    assert 'Install' in html
    assert 'Reject' in html
    assert 'function loadProposals' in js
    assert 'function approveSelectedProposal' in js
    assert 'function installSelectedProposal' in js
    assert 'extractProposalId' in js


def test_tool_proposal_api_list_is_available_for_gui_workflow():
    client = TestClient(app)
    response = client.get('/tool-proposals')
    assert response.status_code == 200
    assert 'tool_proposals' in response.json()
