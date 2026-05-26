TOOL_META={"id":"echo","name":"Echo","description":"Returns the input payload.","version":"0.1.0","input_schema":{"text":"str"},"output_schema":{"text":"str"},"security_level":"SAFE","status":"ACTIVE","module":"tools.echo","function":"run"}
def run(payload: dict) -> dict:
    return {"text": payload.get("text") or payload.get("input") or ""}
