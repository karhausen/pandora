import ast, operator as op
TOOL_META = {"id":"calculator","name":"Calculator","description":"Safely evaluates arithmetic expressions.","version":"0.1.0","input_schema":{"expression":"str"},"output_schema":{"result":"number"},"security_level":"SAFE","status":"ACTIVE","module":"tools.calculator","function":"run"}
OPS = {ast.Add:op.add, ast.Sub:op.sub, ast.Mult:op.mul, ast.Div:op.truediv, ast.Pow:op.pow, ast.USub:op.neg}
def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int,float)): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS: return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS: return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")
def run(payload: dict) -> dict:
    return {"result": _eval(ast.parse(payload["expression"], mode="eval").body)}
