from __future__ import annotations

from pathlib import Path

from core.tool_registry import ToolMeta, ToolRegistry


def register_builtins(tool_dir: Path) -> None:
    registry = ToolRegistry(tool_dir)
    registry.initialize()
    registry.register(
        ToolMeta(
            id="builtin.calculator",
            name="calculator",
            description="Safe calculator for basic arithmetic expressions.",
            input_schema={"expression": "str"},
            output_schema={"expression": "str", "result": "number"},
            safety_level="low",
            test_status="passed",
            module=str(tool_dir / "calculator.py"),
        )
    )


if __name__ == "__main__":
    register_builtins(Path(__file__).resolve().parent)
