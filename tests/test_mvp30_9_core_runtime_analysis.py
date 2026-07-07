from pathlib import Path
import json


def test_mvp30_9_analysis_artifacts_exist():
    assert Path('scripts/core_runtime_analyze.py').exists()
    assert Path('docs/core_runtime_analysis_mvp30_9.md').exists()
    assert Path('docs/core_runtime_analysis_mvp30_9.json').exists()
    assert Path('release/RELEASE_MVP_30_9.md').exists()


def test_mvp30_9_report_is_analyze_only():
    report = json.loads(Path('docs/core_runtime_analysis_mvp30_9.json').read_text(encoding='utf-8'))
    assert report['kind'] == 'mvp30_9_static_core_runtime_analysis'
    assert 'main' in report['entrypoints']
    assert 'core.api' in report['entrypoints']
    assert report['counts']['core_modules_total'] > 0
    assert report['counts']['reachable_core_modules'] > 0
    assert any('do not delete' in rule.lower() for rule in report['rules'])
