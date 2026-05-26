from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    registry.register(
        ToolMeta(
            id="builtin.echo",
            name="echo",
            description="Returns the provided payload for smoke tests and debugging.",
            input_schema={"task": "str"},
            output_schema={"echo": "any"},
            safety_level="low",
            test_status="passed",
            module=str(tool_dir / "echo.py"),
        )
    )


if __name__ == "__main__":
    register_builtins(Path(__file__).resolve().parent)
    print("Built-in tools registered.")
