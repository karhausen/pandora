from __future__ import annotations

from core.models import SecurityLevel, ToolSpec
from core.tool_generator import ToolGenerator
from core.tool_test_generator import ToolTestGenerator


def test_generic_word_counter_code_matches_output_schema():
    spec = ToolSpec(
        id="word_counter",
        name="Word Counter",
        description="Counts the number of words in a given text input",
        capability="word_count",
        input_schema={"text": "string"},
        output_schema={"count": "integer"},
        security_level=SecurityLevel.SAFE,
    )

    namespace: dict = {}
    exec(ToolGenerator().generate_code(spec), namespace)

    result = namespace["run"]({"text": "eins zwei drei"})

    assert result == {"count": 3}


def test_generic_word_counter_test_checks_count_not_only_dict():
    spec = ToolSpec(
        id="word_counter",
        name="Word Counter",
        description="Counts the number of words in a given text input",
        capability="word_count",
        input_schema={"text": "string"},
        output_schema={"count": "integer"},
        security_level=SecurityLevel.SAFE,
    )

    test_code = ToolTestGenerator().generate_test(spec)

    assert 'result["count"] == 3' in test_code
    assert "assert isinstance(result, dict)" not in test_code
