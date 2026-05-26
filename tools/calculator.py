from __future__ import annotations

import ast
import operator as op
from typing import Any

METADATA = {
    "id": "builtin.calculator",
    "name": "calculator",
    "description": "Sicherer Rechner für einfache arithmetische Ausdrücke ohne eval().",
    "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
    "output_schema": {"type": "object", "properties": {"result": {"type": "number"}}},
    "safety_level": "low",
    "version": "0.2.0",
    "test_status": "tested",
}

_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval(node.operand))
    raise ValueError("Only basic arithmetic is allowed")


def run(payload: dict[str, Any]) -> dict[str, Any]:
    expression = payload.get("expression") or payload.get("task", "")
    expression = expression.replace("berechne", "").replace("rechne", "").replace("calculate", "").strip()
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _eval(tree.body)}
