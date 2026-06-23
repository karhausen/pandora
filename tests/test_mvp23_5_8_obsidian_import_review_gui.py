from __future__ import annotations

from fastapi.testclient import TestClient

from core.api import app
from core.registration_validator import RegistrationValidator


def test_obsidian_import_review_web_routes_exist():
    client = TestClient(app)
    res = client.get('/obsidian-import-review')
    assert res.status_code == 200
    assert 'Obsidian Import Review' in res.text
    assert client.get('/web/obsidian-import-review.js').status_code == 200
    assert client.get('/web/obsidian-import-review.css').status_code == 200


def test_obsidian_import_review_api_dashboard_shape():
    client = TestClient(app)
    res = client.get('/api/obsidian/import-review?limit=5')
    assert res.status_code == 200
    data = res.json()
    assert data['kind'] == 'obsidian_import_review_dashboard'
    assert 'candidates' in data
    assert data['safety']['obsidian_read_only'] is True


def test_registration_validator_accepts_import_review_gui():
    report = RegistrationValidator().validate()
    assert report['ok'] is True
