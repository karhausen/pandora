from fastapi.testclient import TestClient

from core.api import app


def test_capability_explorer_page_and_assets_available():
    client = TestClient(app)
    page = client.get('/capability-explorer')
    assert page.status_code == 200
    assert 'Capability Explorer' in page.text
    assert '/web/capability-explorer.js' in page.text
    assert '/web/capability-explorer.css' in page.text

    js = client.get('/web/capability-explorer.js')
    css = client.get('/web/capability-explorer.css')
    assert js.status_code == 200
    assert 'rebuildGraph' in js.text
    assert css.status_code == 200
    assert 'var(--bg)' in css.text


def test_user_gui_links_capability_explorer():
    client = TestClient(app)
    page = client.get('/')
    assert page.status_code == 200
    assert '/capability-explorer' in page.text


def test_capability_api_rebuild_and_list():
    client = TestClient(app)
    rebuild = client.post('/api/capabilities/rebuild')
    assert rebuild.status_code == 200
    payload = rebuild.json()
    assert payload['kind'] == 'capability_graph'
    assert 'nodes' in payload
    assert 'edges' in payload

    listed = client.get('/api/capabilities?limit=20')
    assert listed.status_code == 200
    list_payload = listed.json()
    assert list_payload['kind'] == 'capability_list'
    assert 'capabilities' in list_payload
