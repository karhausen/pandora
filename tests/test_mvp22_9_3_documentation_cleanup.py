from pathlib import Path


def test_readme_is_current_and_useful():
    text = Path('README.md').read_text(encoding='utf-8')
    assert 'MVP 22.9.3' in text
    assert 'Was Pandora aktuell kann' in text
    assert 'Konfiguration' in text
    assert 'Wichtige CLI-Befehle' in text
    assert 'User Knowledge Base' in text
    assert 'MVP 20.4.1' not in text


def test_docs_have_consolidated_entry_points():
    for name in ['README.md', 'overview.md', 'configuration.md', 'commands.md', 'gui.md', 'knowledge_base.md']:
        path = Path('docs') / name
        assert path.exists(), name
        assert path.stat().st_size > 500, name


def test_no_tiny_placeholder_docs_remain():
    tiny = []
    for path in Path('docs').glob('*.md'):
        if path.name == 'README.md':
            continue
        text = path.read_text(encoding='utf-8').strip()
        if len(text) < 250:
            tiny.append(path.name)
    assert tiny == []
