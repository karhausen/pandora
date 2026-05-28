from __future__ import annotations

from .models import ToolSpec


class ToolTestGenerator:
    def generate_test(self, spec: ToolSpec) -> str:
        if spec.id == "json_pretty":
            return '''from generated_tools.json_pretty import run

def test_json_pretty():
    result = run({"text": "{\"b\":2,\"a\":1}"})
    assert result["text"].startswith("{")
    assert "\n" in result["text"]
'''
        if spec.id == "text_reverse":
            return '''from generated_tools.text_reverse import run

def test_text_reverse():
    assert run({"text": "abc"})["text"] == "cba"
'''
        if spec.id == "word_count":
            return '''from generated_tools.word_count import run

def test_word_count():
    assert run({"text": "eins zwei drei"})["count"] == 3
'''
        if spec.id == "timestamp":
            return '''from generated_tools.timestamp import run

def test_timestamp():
    assert "timestamp" in run({})
'''
        return f'''from generated_tools.{spec.id} import run

def test_{spec.id}():
    result = run({{"text": "hello"}})
    assert isinstance(result, dict)
'''
