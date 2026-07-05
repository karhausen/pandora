from pathlib import Path


def test_cognitive_reasoning_layer_exists_and_is_last_resort_gate():
    source = Path("core/cognitive_reasoning_layer.py").read_text(encoding="utf-8").lower()
    assert "cognitive reasoning layer" in source
    assert "last resort" in source
    assert "existing capability" in source
    assert "never route by keywords" in source


def test_capability_orchestrator_uses_reasoning_layer_as_active_path():
    source = Path("core/capability_orchestrator.py").read_text(encoding="utf-8")
    decide_block = source[source.index("    def decide"):source.index("    def _ask_llm")]
    assert "reasoning_layer.reason" in decide_block
    assert "_ask_llm" not in decide_block
    assert "cognitive_reasoning" in decide_block


def test_snapshot_exposes_python_execution_and_tool_factory_as_last_resort():
    source = Path("core/capability_snapshot.py").read_text(encoding="utf-8")
    assert "python_task_execution" in source
    assert "persistent tool/capability only after existing capabilities are insufficient" in source
    assert "Before creating a new capability proposal" in source
